from logging import root
"""
Train a Deferrable Concept Bottleneck Model on CUB-200-2011.
"""
import argparse
import os
import time

import pytorch_lightning as pl
import torch
import torchvision.models
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision.models.resnet import BasicBlock
import pandas as pd

from scc.data.CUB200.cub_loader import generate_data
from scc.defer_baselines import CBMConsistentVision
from scc.networks import InceptionV3, Resn34
from scc.utils import test_model

# torch configurations
torch.set_float32_matmul_precision("medium")
torch.autograd.set_detect_anomaly(True)


def main(
    emb_size: int = 128,
    n_epochs: int = 100,
    n_experts: int = 1,
    n_tasks: int = 1,
    batch_size: int = 256,
    lr: float = 1e-3,
    folds: int = 3,
    psi_loss: str = "ASM",
    num_workers: int = 16,
    # skip_plot: bool = True,
    seed: int = 42,
    network: str = "resnet34",
    cuda_device: int = 0,
    model_expert_task="oracle",
    pretrained_dataset="IMAGENET",
    expert_concept="human",
    baseline_cbm=None,
    res_dir="resultsTest/cub/",
    pretrained_task="x2y"
):

    # Set device using Pytorch Lightning
    pl.seed_everything(seed)
    concept_human = False if expert_concept == "human" else True
    train_loader, val_loader, test_loader, imbalance = generate_data(root_dir="data/",
        config={'batch_size': batch_size, 'num_workers': num_workers, 'weight_loss': True, "human_oracle": concept_human},
        attr_dir="scc/data/CUB200/",
    )
    # Choose architecture
    model_rep = Resn34(BasicBlock, [3, 4, 6, 3])
    str_name = "Rn34"
    if pretrained_dataset == "IMAGENET":
        rn34 = torchvision.models.resnet34(weights="DEFAULT")
        model_rep.load_state_dict(rn34.state_dict())
    elif pretrained_dataset == "CUB":
        if pretrained_task == "x2y":
            path_model_rep = "models/cub/resnet34_pretrained_x2y_CUB_100epochs_seed42.pt"
            model_rep.fc = torch.nn.Linear(model_rep.fc.in_features, 200)
        else:
            path_model_rep = "models/cub/resnet34_pretrained_x2c_CUB_100epochs_seed42.pt"
        model_rep_loaded = torch.load(path_model_rep)
        model_rep.load_state_dict(model_rep_loaded)

    # NOTE: disable concept weighting
    if imbalance is None:
        weight_concepts = None
    else:
        weight_concepts = torch.cat([torch.ones((len(imbalance),1)).float(),torch.tensor(imbalance).unsqueeze(-1).float()], dim=1).unsqueeze(0)
        print(weight_concepts.shape)
    results = pd.DataFrame()
    for fold in range(folds):
        pl.seed_everything(fold + 42)
        print(f"Fold {fold + 1}/{folds}")
        if baseline_cbm == "cbm":
            models = {
                "CBMConsistent{}-PT_{}-independent".format(str_name, pretrained_dataset):
                    CBMConsistentVision(model_rep=model_rep,
                                        emb_size=emb_size,
                                        n_classes=200,
                                        n_concepts=112,
                                        n_experts=1,
                                        n_tasks=1,
                                        learning_rate=lr,
                                        psi_loss="CE-nohuman",
                                        psi_loss_tasks="CE-nohuman",
                                        psi_loss_concepts="CE-nohuman",
                                        weights_concepts=weight_concepts,
                                        concept_lambdas=[5 for _ in range(112)]
                                        )}
        else:
            models = {}
        if psi_loss in ["CE", "ASM", "ASM", "OVA"]:
            for cost in [
                0,
                .01,
                .02, # values from .02 to .05 added for rebuttal results and used only for DCBM-NoTask
                .0225,
                .0250,
                .0275,
                .03,
                .04,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.30,
                .35,
                .40,
                .45,
                .50
            ]:
                for label_sm in [True]:
                    for tm in ["independent"]:
                        models[
                            "CBMConsistentInV3-PT_{}-{}-{}-cost_{}-LS_{}".format(
                                pretrained_dataset, psi_loss, tm, str(cost), label_sm)
                        ] = CBMConsistentVision(model_rep=model_rep,
                                                emb_size=emb_size,
                                                n_classes=200,
                                                n_concepts=112,
                                                n_tasks=1,
                                                n_experts=1,
                                                cost=cost,
                                                learning_rate=lr,
                                                psi_loss=psi_loss,
                                                psi_loss_tasks=psi_loss,
                                                psi_loss_concepts=psi_loss,
                                                weights_concepts=weight_concepts,
                                                concept_lambdas=[
                                                    5 for _ in range(112)]
                                                )
                        models[
                            "CBMConsistentInV3-PT_{}-{}-{}-cost_{}-LS_{}-NoHumanFinal".format(
                                pretrained_dataset, psi_loss, tm, str(cost), label_sm)
                        ] = CBMConsistentVision(model_rep=model_rep,
                                                emb_size=emb_size,
                                                n_classes=200,
                                                n_concepts=112,
                                                n_tasks=1,
                                                n_experts=1,
                                                cost=cost,
                                                learning_rate=lr,
                                                psi_loss=psi_loss,
                                                psi_loss_tasks="CE-nohuman",
                                                psi_loss_concepts=psi_loss,
                                                weights_concepts=weight_concepts,
                                                concept_lambdas=[
                                                    5 for _ in range(112)]
                                                )
                        models[
                            "CBMConsistentInV3-PT_{}-{}-{}-cost_{}-LS_{}-NoHumanConcepts".format(
                                pretrained_dataset, psi_loss, tm, str(cost), label_sm)
                        ] = CBMConsistentVision(model_rep=model_rep,
                                                emb_size=emb_size,
                                                n_classes=200,
                                                n_concepts=112,
                                                n_tasks=1,
                                                n_experts=1,
                                                cost=cost,
                                                learning_rate=lr,
                                                psi_loss=psi_loss,
                                                psi_loss_concepts="CE-nohuman",
                                                psi_loss_tasks=psi_loss,
                                                weights_concepts=weight_concepts,
                                                concept_lambdas=[
                                                    5 for _ in range(112)]
                                                )
        else:
            raise ValueError(f"Unknown psi_loss: {psi_loss}")
        # define the model
        for model_name, model in models.items():
            print(f"Training {model_name}")

            # train and log the model
            logger = TensorBoardLogger("tb_logs", name=model_name)
            trainer = pl.Trainer(
                max_epochs=n_epochs,
                accelerator="cuda",
                devices=[cuda_device],
                logger=logger,
                check_val_every_n_epoch=5,
                gradient_clip_val=5,
                deterministic=True,
                callbacks=[
                    EarlyStopping(
                        monitor="err_c_y_val",
                        patience=10,
                        mode="min",
                        verbose=False,
                        min_delta=0.0,
                    )]
                # type: ignore
            )
            # compute training time
            start = time.time()
            path_model = "models/cub/check{}__{}_{}__CE_default_f{}_ES__{}.pt".format(
                model_name, network, model_expert_task, 42 + fold, emb_size)
            if weight_concepts is not None:
                path_model = "{}.pt".format(path_model.replace(".pt", "_Weighted.pt"))
            if pretrained_task == "x2y":
                path_model = path_model.replace("CE_default", "CE_default__x2y")
            print(path_model)
            model.to("cuda:{}".format(cuda_device))
            trainer.fit(model, train_loader, val_loader)
            os.makedirs("models/cub", exist_ok=True)
            train_time = time.time() - start
            model.training_time = train_time
            torch.save(model.state_dict(), path_model)
            print(f"Training time: {train_time:.2f}")
            model.eval()
            print("Model loaded")
            # print(f"Training time: {train_time:.2f}")
            tmp = test_model(testloader=test_loader, model=model, seed=fold + seed, path_results="resultsCUB.csv")
            tmp["full_name"] = model_name
            # tmp["expert_accuracy_concept"] = baseline_acc_concept
            tmp["n_concepts"] = 112
            tmp["n_experts"] = n_experts
            tmp["n_tasks"] = n_tasks
            tmp["n_samples"] = test_loader.dataset.__len__()
            tmp["n_epochs"] = n_epochs
            tmp["train_time"] = train_time
            results = pd.concat([results, tmp], axis=0)
            print("\n \n \n \n \n\n\n\n Accuracy task is {}".format(tmp["acc_test_task"].values[0]))
            os.makedirs(res_dir, exist_ok=True)
            res_path = res_dir+ "/NEURIPS2025_ALLresults{}wWeights_{}-humanTask{}_{}.csv".format(
                    pretrained_dataset,
                    psi_loss,
                    model_expert_task,
                network
                )
            if weight_concepts is not None:
                res_path = res_path.replace(".csv", "_Weighted.csv")
            results.to_csv(
                res_path
                ,
                index=False,
            )
            print("just printed results")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--psi_loss", type=str, default="ASM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_device", type=int, default=0)
    parser.add_argument("--expert_task", type=str, default="oracle")
    parser.add_argument("--expert_concept", type=str, default="human")
    parser.add_argument("--pretrained", type=str, default="CUB")
    parser.add_argument("--baseline_cbm", type=str, default=None)
    parser.add_argument("--network", type=str, default="resnet34")
    parser.add_argument("--pretrained_task", type=str, default="x2y")
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
        # config_str=args.all_settings,
    )
