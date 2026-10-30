"""
Run the DCBM deferral-cost sweep (see DCBM.py)
"""
import argparse
import os
import time

import pytorch_lightning as pl
import torch
import torchvision.models
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision.models.resnet import BasicBlock
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd

from scc.data.animals.animals_loader import generate_data
from scc.networks import Resn34
from scc.utils import test_model
from DCBM import DCBM

# torch configurations
torch.set_float32_matmul_precision("medium")

COST_GRID = [0.0, 0.01, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.04,
             0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def _make_backbone(pretrained_dataset, n_classes, pretrained_task):
    """Build a *fresh* encoder for every model.

    Each DCBM must start from the pretrained weights, not from the fine-tuned
    state left behind by the previous model in the sweep -- otherwise the cost
    grid is trained sequentially on top of itself and no two points in the
    sweep are comparable.
    """
    backbone = Resn34(BasicBlock, [3, 4, 6, 3])
    if pretrained_dataset == "IMAGENET":
        backbone.load_state_dict(torchvision.models.resnet34(weights="DEFAULT").state_dict())
    elif pretrained_dataset == "CUB":
        if pretrained_task == "x2y":
            path_model_rep = "models/cub/resnet34_pretrained_x2y_CUB_100epochs_seed42.pt"
            backbone.fc = torch.nn.Linear(backbone.fc.in_features, n_classes)
        else:
            path_model_rep = "models/cub/resnet34_pretrained_x2c_CUB_100epochs_seed42.pt"

        if not os.path.exists(path_model_rep):
            raise FileNotFoundError(
                f"{path_model_rep} not found. Pretrain the encoder first, "
                f"or run with --pretrained IMAGENET."
            )

        backbone.load_state_dict(torch.load(path_model_rep))
    return backbone


def _dcbm_kwargs(embedding_size, lr, cost, psi_loss,
                 psi_loss_tasks, psi_loss_concepts, weight_concepts,
                 label_sm, tm, n_classes, n_concepts):
    return dict(
        embedding_size=embedding_size,
        n_classes=n_classes,
        n_concepts=n_concepts,
        n_tasks=1,
        n_experts=1,
        deferral_cost=cost,
        learning_rate=lr,
        psi_loss=psi_loss,
        psi_loss_tasks=psi_loss_tasks,
        psi_loss_concepts=psi_loss_concepts,
        weights_concepts=weight_concepts,
        concept_lambdas=[5 for _ in range(n_concepts)],
        label_smoothing=label_sm,
        training_mode=tm,
        pretrain_freeze=True,
    )


def main(
    embedding_size: int = 128,
    n_epochs: int = 100,
    n_experts: int = 1,
    n_tasks: int = 1,
    batch_size: int = 256,
    lr: float = 1e-3,
    folds: int = 3,
    psi_loss: str = "ASM",
    num_workers: int = 0,
    seed: int = 42,
    network: str = "resnet34",
    cuda_device: int = 0,
    model_expert_task="oracle",
    pretrained_dataset="IMAGENET",
    expert_concept="human",
    baseline_cbm=None,
    res_dir="resultsTest/dcbm/",
    pretrained_task="x2y",
    expert_concept_acc: float = 1.0,
    expert_task_acc: float = 1.0,
):

    run_id = time.strftime("%Y%m%d-%H%M%S")

    # Set device using Pytorch Lightning
    pl.seed_everything(seed)
    concept_human = False if expert_concept == "human" else True
    data_config = {'batch_size': batch_size, 'num_workers': num_workers,
                   'weight_loss': True, "human_oracle": concept_human,
                   'expert_concept_accuracy': expert_concept_acc,
                   'expert_task_accuracy': expert_task_acc,
                   'expert_seed': seed}
    # Choose architecture
    str_name = "Rn34"

    results = pd.DataFrame()
    os.makedirs(res_dir, exist_ok=True)
    for fold in range(folds):
        pl.seed_everything(fold + seed)
        print(f"Fold {fold + 1}/{folds}")

        # Resample the split per fold so error bars reflect sampling variance,
        # not just weight initialisation. Everything downstream of the split
        # (class counts, concept imbalance) is therefore fold-local.
        train_loader, val_loader, test_loader, imbalance = generate_data(
            root_dir="data/", config=data_config, seed=fold + seed,
        )
        n_classes = train_loader.dataset.n_classes
        n_concepts = train_loader.dataset.n_concepts

        # Per-concept class weights: 1 for negatives, neg/pos ratio for positives
        if imbalance is None:
            weight_concepts = None
        else:
            weight_concepts = torch.cat(
                [torch.ones((len(imbalance), 1)).float(),
                 torch.tensor(imbalance).unsqueeze(-1).float()], dim=1).unsqueeze(0)
            print(weight_concepts.shape)

        label_sm = True
        tm = "independent"

        models = {}
        if baseline_cbm == "cbm":
            models["CBM-PT_{}-{}-LS_{}".format(pretrained_dataset, tm, label_sm)] = _dcbm_kwargs(
                embedding_size=embedding_size,
                lr=lr,
                cost=0.0,
                psi_loss="CE-nohuman",
                psi_loss_tasks="CE-nohuman",
                psi_loss_concepts="CE-nohuman",
                weight_concepts=weight_concepts,
                n_classes=n_classes,
                n_concepts=n_concepts,
                label_sm=label_sm,
                tm=tm,
            )
        if psi_loss not in ["CE", "ASM", "OVA"]:
            raise ValueError(f"Unknown psi_loss: {psi_loss}")

        variants = [
            ("",                   psi_loss,      psi_loss),
            ("-NoHumanFinal",      "CE-nohuman",  psi_loss),
            ("-NoHumanConcepts",   psi_loss,      "CE-nohuman"),
        ]
        for cost in COST_GRID:
            base_name = "DCBM-PT_{}-{}-{}-cost_{}-LS_{}".format(
                pretrained_dataset, psi_loss, tm, str(cost), label_sm)
            for suffix, ploss_tasks, ploss_concepts in variants:
                models[base_name + suffix] = _dcbm_kwargs(
                    embedding_size=embedding_size,
                    lr=lr,
                    cost=cost,
                    psi_loss=psi_loss,
                    psi_loss_tasks=ploss_tasks,
                    psi_loss_concepts=ploss_concepts,
                    weight_concepts=weight_concepts,
                    n_classes=n_classes,
                    n_concepts=n_concepts,
                    label_sm=label_sm,
                    tm=tm,
                )
        # define the model
        for model_name, model_kwargs in models.items():
            print(f"Training {model_name}")

            # Fresh encoder per model (see _make_backbone)
            model = DCBM(
                model_backbone=_make_backbone(pretrained_dataset, n_classes, pretrained_task),
                **model_kwargs,
            )

            # train and log the model
            logger = TensorBoardLogger("tb_logs", name=model_name)
            use_cuda = torch.cuda.is_available()

            checkpoint = ModelCheckpoint(monitor="loss_val", mode="min",
                                         save_top_k=1, save_weights_only=True)

            trainer = pl.Trainer(
                max_epochs=n_epochs,
                accelerator="gpu" if use_cuda else "cpu",
                devices=[cuda_device] if use_cuda else 1,
                logger=logger,
                check_val_every_n_epoch=1,
                gradient_clip_val=5,
                deterministic=True,
                log_every_n_steps=1,
                callbacks=[
                    EarlyStopping(
                        monitor="loss_val",
                        patience=10,
                        mode="min",
                        verbose=False,
                        min_delta=0.0,
                    ),
                    checkpoint,]
            )
            # compute training time
            start = time.time()
            model_dir = os.path.join("models", "animals")
            os.makedirs(model_dir, exist_ok=True)
            path_model = os.path.join(
                model_dir,
                "{}__eC{}_eT{}_f{}_emb{}{}.pt".format(
                    model_name, expert_concept_acc, expert_task_acc, seed + fold, embedding_size,
                    "_Weighted" if weight_concepts is not None else "",
                ),
            )
            print(path_model)
            trainer.fit(model, train_loader, val_loader)
            train_time = time.time() - start

            best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(best["state_dict"])
            os.remove(checkpoint.best_model_path)

            model.training_time = train_time
            torch.save(model.state_dict(), path_model)
            print(f"Training time: {train_time:.2f}")
            model.eval()
            print(f"Best model loaded (epoch {best['epoch']})")
            tmp = test_model(testloader=test_loader, model=model, seed=fold + seed,
                             path_results=os.path.join(res_dir, f"resultsDCBM_{run_id}.csv"))
            tmp["full_name"] = model_name
            tmp["n_concepts"] = n_concepts
            tmp["n_experts"] = n_experts
            tmp["n_tasks"] = n_tasks
            tmp["n_samples"] = test_loader.dataset.__len__()
            tmp["n_epochs"] = n_epochs
            tmp["train_time"] = train_time
            tmp["fold"] = fold
            tmp["expert_concept_acc"] = expert_concept_acc
            tmp["expert_task_acc"] = expert_task_acc
            tmp["run_id"] = run_id
            results = pd.concat([results, tmp], axis=0)
            print("\n \n \n \n \n\n\n\n Accuracy task is {}".format(tmp["acc_test_task"].values[0]))
            res_path = os.path.join(
                res_dir,
                f"DCBM_{pretrained_dataset}_{psi_loss}_eC{expert_concept_acc}_eT{expert_task_acc}.csv",
            )
            if weight_concepts is not None:
                res_path = res_path.replace(".csv", "_Weighted.csv")
            results.to_csv(
                res_path,
                index=False,
            )
            print("just printed results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--psi_loss", type=str, default="ASM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_device", type=int, default=0)
    parser.add_argument("--expert_task", type=str, default="oracle")
    parser.add_argument("--expert_concept", type=str, default="human")
    parser.add_argument("--pretrained", type=str, default="IMAGENET")
    parser.add_argument("--baseline_cbm", type=str, default=None)
    parser.add_argument("--network", type=str, default="resnet34")
    parser.add_argument("--pretrained_task", type=str, default="x2y")
    # Expert competence. 1.0 = oracle, which leaves the l_2 term of the DCBM
    parser.add_argument("--expert_concept_acc", type=float, default=0.8)
    parser.add_argument("--expert_task_acc", type=float, default=0.8)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    main(
        psi_loss=args.psi_loss,
        seed=args.seed,
        folds=args.folds,
        cuda_device=args.n_device,
        pretrained_dataset=args.pretrained,
        baseline_cbm=args.baseline_cbm,
        network=args.network,
        pretrained_task=args.pretrained_task,
        expert_concept=args.expert_concept,
        expert_concept_acc=args.expert_concept_acc,
        expert_task_acc=args.expert_task_acc,
        num_workers=args.num_workers
    )
