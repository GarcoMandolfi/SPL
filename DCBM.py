from typing import Any

import torch
import pytorch_lightning as pl
from torch.optim import Adam
import torchmetrics
from scc.networks import WideResNet, Resn34
import torchvision.models as models

losses = ["CE", "OVA", "ASM", "RS", "CE-nohuman"]

class DCBM(pl.LightningModule):

    def __init__(
        self,
        model_backbone: torch.nn.Module,
        embedding_size: int,
        n_classes: int,
        n_concepts: int,
        n_tasks: int,
        n_experts: int,
        psi_loss: str = "CE",
        label_smoothing: bool = False,
        deferral_cost: float = 0.5,
        learning_rate: float = 1e-2,
        training_mode: str = "independent",
        concept_lambdas: list = None,
        task_lambdas: list = None,
        epsilon: float = 1e-8,
        alpha: float = 0,
        psi_loss_concepts=None,
        psi_loss_tasks=None,
        weights_concepts=None,
        weights_tasks=None,
        optimizer: str = "AdamW",
        halved: bool = False,
        setup_optimizer: str = "cifar10",
        pretrain_freeze: bool = True,
    ):
        
        super(DCBM, self).__init__()

        self.n_concepts = n_concepts    # All binary
        self.n_classes = n_classes      # Same for all final tasks
        self.n_tasks = n_tasks
        self.n_experts = n_experts

        if psi_loss not in losses:
            raise ValueError(f"Invalid psi_loss: {psi_loss}. Must be one of {losses}.")
        self.psi_loss = psi_loss
        self.psi_loss_tasks = psi_loss if psi_loss_tasks is None or psi_loss_tasks not in losses else psi_loss_tasks
        self.psi_loss_concepts = psi_loss if psi_loss_concepts is None or psi_loss_concepts not in losses else psi_loss_concepts

        self.label_smoothing_free = label_smoothing
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.deferral_cost = deferral_cost

        self.task_accuracy_train = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.n_classes)
        self.concept_accuracy_train = torchmetrics.classification.Accuracy(task="multilabel", num_labels=self.n_concepts)

        self.task_accuracy_val = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.n_classes)
        self.concept_accuracy_val = torchmetrics.classification.Accuracy(task="multilabel", num_labels=self.n_concepts)

        self.epsilon = 1e-5 if self.psi_loss == "ASM" and epsilon < 1e-5 else epsilon
        self.eps = self.epsilon
        self.alpha = alpha

        self.concept_lambdas = torch.ones(size=(1, n_concepts, 1), device=self.device)
        tmp = [1 for _ in range(n_concepts)] if concept_lambdas is None else concept_lambdas
        self.concept_lambdas[0, :, 0] = torch.tensor(tmp, device=self.device)
        
        self.task_lambdas = torch.ones(size=(1, n_tasks, 1), device=self.device)
        tmp = [1 for _ in range(n_tasks)] if task_lambdas is None else task_lambdas
        self.task_lambdas[0, :, 0] = torch.tensor(tmp, device=self.device)

        if self.n_experts not in [0, 1]:
            raise NotImplementedError("Only one or zero experts are supported.")
        assert deferral_cost >= 0 and deferral_cost <= 1

        self.weights_concepts = weights_concepts
        self.weights_tasks = weights_tasks

        n_outputs = n_classes + n_experts   # For each task head, 2 outputs plus its expert
        self.n_outputs_concepts = (2 + n_experts) * n_concepts
        self.n_outputs_tasks = (n_outputs) * n_tasks

        self.model_backbone = model_backbone
        self.pretrain_freeze = pretrain_freeze
        if self.pretrain_freeze:
            for param in self.model_backbone.parameters():
                param.requires_grad = False

        assert hasattr(self.model_backbone, "repr")
        if type(model_backbone) == WideResNet:
            self.in_features = self.model_backbone.fc2.in_features
        elif type(model_backbone) == Resn34:
            self.in_features = self.model_backbone.fc.in_features

        if halved:
            self.hidden_layer_out = embedding_size/2
        else:
            self.hidden_layer_out = embedding_size
        
        self.model_concept = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.in_features, embedding_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(embedding_size, int(self.hidden_layer_out)),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(int(self.hidden_layer_out), 2 + n_experts),
                )
                for _ in range(n_concepts)
            ]
        )

        self.model_task = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(n_concepts, embedding_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(embedding_size, int(self.hidden_layer_out)),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(int(self.hidden_layer_out), n_outputs),
                )
                for _ in range(n_tasks)
            ]
        )

        self.training_mode = training_mode
        self.err_c_y_val = []
        self.err_rate_task_val = []
        self.setup_optimizer = setup_optimizer
    
    def repr(self, x_batch: torch.Tensor):
        return self.model_backbone.repr(x_batch)
    
    def configure_optimizers(self):
        if (self.optimizer.lower() == "adam") and (self.setup_optimizer != "cub"):
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr = self.learning_rate,
                weight_decay = 4e-5,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, verbose=True)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        elif self.setup_optimizer == "cub":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=.90,
                weight_decay=4e-5,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                verbose=True,
                patience=10,
                factor=0.1,
                min_lr=getattr(self, 'min_lr', 1e-5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",
                }
            }
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr = self.learning_rate,
                weight_decay = 4e-5,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, verbose=True)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        
    def lr_scheduler_step(self, scheduler, metric=None):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is None:
                raise ValueError("Metric must be provided for ReduceLROnPlateau scheduler.")
            scheduler.step(metric)
        else:
            scheduler.step()
    
    def concept_forward(self, x_batch: torch.Tensor):
        c_logit = []
        x_batch = self.repr(x_batch)
        for module in self.model_concept:
            c_logit.append(module.forward(x_batch))
        return torch.stack(c_logit, dim=1)
    
    def task_forward(self, c_batch: torch.Tensor):
        y_logit = []
        for module in self.model_task:
            y_logit.append(module.forward(c_batch))
        return torch.stack(y_logit, dim=1)
    
    def _vector_loss(self,
                     y_logit: torch.Tensor,    # Model logits per sample, per task/concept head
                     y_batch: torch.Tensor,     # Target lablels
                     y_expert: torch.Tensor,    # Expert-provided labels
                     concepts: bool = False,):
        
        assert len(y_batch.size()) == 3
        assert len(y_expert.size()) == 3
        assert len(y_logit.size()) == 3

        if concepts:
            self.indicator = 2
            lambdas = self.concept_lambdas.to(self.device)
            active_loss = self.psi_loss_concepts

            if self.weights_concepts is None:
                w_vec = torch.tensor(1).to(self.device)
            else:
                assert self.weights_concepts.size() == (1, self.n_concepts, self.indicator)
                weights = self.weights_concepts.to(self.device)
                w_batch = weights.expand(y_batch.shape[0], self.n_concepts, self.indicator)
                w_vec = torch.gather(w_batch, -1, y_batch)
        
        else:
            self.indicator = self.n_classes
            lambdas = self.task_lambdas.to(self.device)
            active_loss = self.psi_loss_tasks
            if self.weights_tasks is None:
                w_vec = torch.tensor(1).to(self.device)
            else:
                assert self.weights_tasks.size() == (1, self.n_tasks, self.indicator)
                weights = self.weights_tasks.to(self.device)
                w_batch = weights.expand(y_batch.shape[0], self.n_tasks, self.indicator)
                w_vec = torch.gather(w_batch, -1, y_batch)

        mask_right = y_expert == y_batch
        mask_wrong = y_expert != y_batch

        if active_loss == "CE":
            # first term of the loss by Liu et al. 2024 (no label smoothing)
            l_1_all =  (-torch.log_softmax(y_logit, dim=-1))
            l_1 = torch.gather(l_1_all, -1, y_batch)
            #weight = self.weights # tensor 1 * n_concepts l_1 *weight

            # second term for the loss function defined by Liu (still label smoothing)
            if self.label_smoothing_free:
                l_2 = +(self.deferral_cost) * (torch.min(l_1_all[: , : , : self.indicator], dim = -1)).values.unsqueeze(-1) * mask_wrong.to(torch.float32)
            else:
                l_2 = +(self.deferral_cost) * (torch.sum(l_1_all[: , : , : self.indicator], dim = -1)).unsqueeze(-1) * mask_wrong.to(torch.float32)

            # third term for the loss function defined by Liu (no label smoothing)
            l_3 = +(1 - self.deferral_cost) * mask_right.to(torch.float32) * l_1_all[: , : , self.indicator:]

            loss = torch.mean(l_1 * w_vec * lambdas) + torch.mean(l_2 * lambdas) + torch.mean(l_3 * w_vec * lambdas)
        
        elif active_loss == "RS" :
            exp_1_all = torch.exp(y_logit)
            exp_gt = torch.gather(exp_1_all, -1, y_batch)
            exp_humans = exp_1_all[: , : , self.indicator :]

            loss_vector = -2 * torch.log((exp_gt + exp_humans * mask_right.to(torch.float32)) / torch.sum(exp_1_all, dim=-1).unsqueeze(-1) + self.eps) * w_vec
            loss = (1 - self.alpha) * torch.mean(loss_vector)

            if self.alpha > 0:
                l_1_all = -torch.log_softmax(y_logit[:, :, : self.indicator], dim=-1)
                l_1 = torch.gather(l_1_all, -1, y_batch) * w_vec
                loss += self.alpha * torch.mean(l_1 * lambdas)
            
            if self.label_smoothing_free:
                raise NotImplementedError("Label smoothing is not implemented for RS loss.")

        elif active_loss == "OVA":
            phi_pos = -torch.nn.functional.logsigmoid(y_logit)    # φ(g_k)   -> push head k up
            phi_neg = -torch.nn.functional.logsigmoid(-y_logit)   # φ(-g_k)  -> push head k down
            sum_phi_neg = torch.sum(phi_neg, dim=-1).unsqueeze(-1)

            # L_OVA(g, y) = φ(g_y) + Σ_{k≠y} φ(-g_k)
            l_1 = (torch.gather(phi_pos, -1, y_batch)
                + sum_phi_neg
                - torch.gather(phi_neg, -1, y_batch))

            if self.label_smoothing_free:
                # c · min_k L_OVA(g, k)  over the model's own classes
                l_2 = self.deferral_cost * mask_wrong.to(torch.float32) * torch.min(
                    phi_pos[:, :, :self.indicator]
                    + sum_phi_neg
                    - phi_neg[:, :, :self.indicator],
                    dim=-1,
                ).values.unsqueeze(-1)
            else:
                l_2 = self.deferral_cost * mask_wrong.to(torch.float32) * (
                    (torch.sum(phi_pos[:, :, :self.indicator], dim=-1)
                    + torch.sum(phi_neg[:, :, :self.indicator], dim=-1))
                    * (self.indicator - self.n_experts)
                    + torch.sum(phi_neg[:, :, self.indicator:], dim=-1) * self.n_experts
                ).unsqueeze(-1)

            # flip the defer head from "down" to "up" when the expert was right
            l_3 = (1 - self.deferral_cost) * mask_right.to(torch.float32) * torch.sum(
                phi_pos[:, :, self.indicator:] - phi_neg[:, :, self.indicator:],
                dim=-1,
            ).unsqueeze(-1)

            loss = (torch.mean(l_1 * w_vec * lambdas)
                    + torch.mean(l_2 * lambdas)
                    + torch.mean(l_3 * w_vec * lambdas))

        elif active_loss == "ASM":
            # shift so the largest logit is 0 -> every exp() lands in (0, 1]
            y_logit = y_logit - torch.max(y_logit, dim=-1).values.unsqueeze(-1)

            nll_ml = -torch.nn.functional.log_softmax(y_logit[:, :, :self.indicator], dim=-1)
            nll_ml_gt = torch.gather(nll_ml, -1, y_batch)

            exp_all = torch.exp(y_logit)
            exp_ml = exp_all[:, :, :self.indicator]
            exp_expert = exp_all[:, :, self.indicator:]

            s_ml = torch.sum(exp_ml, dim=-1).unsqueeze(-1)
            m_ml = torch.max(exp_ml, dim=-1).values.unsqueeze(-1)

            num = s_ml - m_ml                                                    # runner-up mass
            den = torch.sum(exp_all, dim=-1).unsqueeze(-1) - m_ml + self.eps
            log_p_predict = torch.log(num / den + self.eps)                      # log P(not defer)

            l_1 = nll_ml_gt - log_p_predict

            if self.label_smoothing_free:
                l_2 = self.deferral_cost * mask_wrong.to(torch.float32) * torch.min(
                    nll_ml - torch.log(torch.clamp(1 - exp_expert / den, min=self.eps)),
                    dim=-1,
                ).values.unsqueeze(-1)
            else:
                l_2 = self.deferral_cost * mask_wrong.to(torch.float32) * (
                    torch.sum(nll_ml, dim=-1).unsqueeze(-1) - self.indicator * log_p_predict
                )

            l_3 = (1 - self.deferral_cost) * mask_right.to(torch.float32) * (
                -torch.log(torch.clamp(exp_expert / den, min=self.eps))
                + torch.log(torch.clamp(num / den, min=self.eps))
            )

            loss = (torch.mean(l_1 * w_vec * lambdas)
                    + torch.mean(l_2 * lambdas)
                    + torch.mean(l_3 * w_vec * lambdas))

        elif active_loss == "CE-nohuman":
            # Table 1's CE, restricted to the model's own classes: the defer slot is
            # excluded from the softmax, so it receives no gradient at all.
            l_1_all = -torch.log_softmax(y_logit[:, :, :self.indicator], dim=-1)
            l_1 = torch.gather(l_1_all, -1, y_batch)
            loss = torch.mean(w_vec * l_1 * lambdas)

        else:
            raise NotImplementedError("Loss not yet implemented.")
        
        return loss
    
    def forward(self, x_batch: torch.Tensor, y_expert: torch.Tensor | None = None) -> torch.Tensor:
        c_logit = self.concept_forward(x_batch)
        c_logit = c_logit.view(-1, self.n_concepts, 2 + self.n_experts)

        if (y_expert is None) or (self.psi_loss == "CE-nohuman"):
            return c_logit

        # Route through predict_from_logits: the defer slot of a concept head is
        # index 2, not n_classes, and task_forward needs floats.
        c_preds = self.predict_from_logits(c_logit, y_expert, concepts=True)
        return self.task_forward(c_preds.view(-1, self.n_concepts).float())
    
    def predict_from_logits(self, y_logit: torch.Tensor, y_expert: torch.Tensor | None = None, concepts = False) -> torch.Tensor:
        if concepts:
            index_expert = 2
        else:
            index_expert = self.n_classes

        # Compute the predictions
        if (self.psi_loss_concepts == "CE-nohuman") and (concepts is True):
            preds = torch.argmax(y_logit[:,:,:index_expert], dim=-1).unsqueeze(-1)
        elif (self.psi_loss_tasks == "CE-nohuman") and (concepts is False):
            preds = torch.argmax(y_logit[:,:,:index_expert], dim=-1).unsqueeze(-1)
        else:
            preds = torch.argmax(y_logit, dim=-1).unsqueeze(-1)
            if self.n_experts > 0:
                assert self.n_experts == 1
                assert y_expert is not None
                # Mask for index of expert
                mask_expert = preds == index_expert
                # Replace the expert prediction with the expert label
                preds = torch.where(mask_expert, y_expert, preds)
        return preds
    
    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        x_batch = batch[0]
        y_batch = batch[1]
        y_expert = batch[2] if self.n_experts > 0 else None
        c_batch = batch[3]
        c_expert = batch[4] if self.n_experts > 0 else None

        if self.n_tasks == 1:
            y_batch = y_batch.view(-1, self.n_tasks, 1)
        c_batch = c_batch.unsqueeze(-1)
        # The loader yields (B, n_concepts); the sequential/joint paths below
        # combine c_expert with (B, n_concepts, 1) tensors.
        if c_expert is not None and c_expert.dim() == 2:
            c_expert = c_expert.unsqueeze(-1)

        #Compute concept logits
        c_logit = self.concept_forward(x_batch)
        c_logit = c_logit.view(-1, self.n_concepts, 2 + self.n_experts)

        #Compute task logits
        if self.training_mode == "independent":
            y_logit = self.task_forward(c_batch.float().squeeze(-1))
        elif self.training_mode == "sequential":
            c_preds = self.predict_from_logits(c_logit, y_expert=c_expert, concepts=True)
            y_logit = self.task_forward(c_preds.float().squeeze(-1))
        elif self.training_mode == "joint":
            c_softmax = torch.nn.functional.softmax(c_logit[: , : , :2], dim = -1)
            c_sigmoid = torch.sigmoid(c_logit[: , : , 2:])
            c_preds = c_softmax[: , : , 1].unsqueeze(-1) * (1 - c_sigmoid) + c_sigmoid * c_expert
            y_logit = self.task_forward(c_preds.float().squeeze(-1))
        
        if self.n_experts > 0:
            y_expert = y_expert.view(-1, self.n_tasks, self.n_experts)
            c_expert = c_expert.view(-1, self.n_concepts, self.n_experts)
            concept_loss = self._vector_loss(c_logit, c_batch, c_expert, concepts=True)
            task_loss = self._vector_loss(y_logit, y_batch, y_expert, concepts=False)
            loss = concept_loss + task_loss
        else:
            raise NotImplementedError("Only one expert is supported. If you want the baseline no expert use psi_loss == 'CE-nohuman'")

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}
    
    def _run_step(self, batch: torch.Tensor):
        x_te = batch[0]
        y_te = batch[1]
        y_human = batch[2]
        c_te = batch[3]
        c_human = batch[4]

        c_logit = self.concept_forward(x_te)
        c_logit = c_logit.view(-1, self.n_concepts, 2 + self.n_experts)

        if len(c_human.shape) == 2:
            c_human = c_human.unsqueeze(-1)
        if self.n_experts == 0:
            c_pred = self.predict_from_logits(c_logit, None)
        else:
            c_pred = self.predict_from_logits(c_logit, c_human, concepts=True)

        y_logit = self.task_forward(c_pred.view(-1, self.n_concepts).float())

        if self.n_experts == 0:
            y_pred = self.predict_from_logits(y_logit, None, concepts=False)
        else:
            y_pred = self.predict_from_logits(y_logit, y_human.view(-1, self.n_tasks, self.n_experts))
        
        y_preds = y_pred.view(-1, self.n_tasks)
        c_preds = c_pred.view(-1, self.n_concepts)
        return y_preds, c_preds, c_logit, y_logit, c_human, y_human, c_te, y_te
    
    def validation_step(self, batch: torch.Tensor):
        x_batch = batch[0]
        y_batch = batch[1]
        y_expert = batch[2] if self.n_experts > 0 else None
        c_batch = batch[3]
        c_expert = batch[4] if self.n_experts > 0 else None

        if self.n_tasks == 1:
            y_batch = y_batch.view(-1, self.n_tasks, 1)
        c_batch = c_batch.unsqueeze(-1)
        # See training_step: normalise c_expert to (B, n_concepts, 1).
        if c_expert is not None and c_expert.dim() == 2:
            c_expert = c_expert.unsqueeze(-1)

        #Compute concept logits
        c_logit = self.concept_forward(x_batch)
        c_logit = c_logit.view(-1, self.n_concepts, 2 + self.n_experts)

        #Comupte task logits
        if self.training_mode == "independent":
            y_logit = self.task_forward(c_batch.float().squeeze(-1))
        elif self.training_mode == "sequential":
            c_preds = self.predict_from_logits(c_logit, y_expert=c_expert, concepts=True)
            y_logit = self.task_forward(c_preds.float().squeeze(-1))
        elif self.training_mode == "joint":
            c_softmax = torch.nn.functional.softmax(c_logit[:, :, :2], dim=-1)
            c_sigmoids = torch.sigmoid(c_logit[:, :, 2:])
            c_preds = (
                c_softmax[:, :, 1].unsqueeze(-1) * (1 - c_sigmoids)
                + (c_sigmoids) * c_expert
            )
            y_logit = self.task_forward(c_preds.float().squeeze(-1))
        
        if self.n_experts > 0:
            y_expert = y_expert.view(-1, self.n_tasks, self.n_experts)
            c_expert = c_expert.view(-1, self.n_concepts, self.n_experts)
            concept_loss = self._vector_loss(c_logit, c_batch, c_expert, concepts=True)
            task_loss = self._vector_loss(y_logit, y_batch, y_expert, concepts=False)
            loss = concept_loss + task_loss
        
        else:
            raise NotImplementedError("Only one expert is supported. If you want the baseline no expert use psi_loss == 'CE-nohuman'")
        self.log("loss_val", loss)

        
        c_preds = self.predict_from_logits(c_logit, c_expert, concepts=True)
        y_logit_pipe = self.task_forward(c_preds.view(-1, self.n_concepts).float())
        y_preds = self.predict_from_logits(y_logit_pipe, y_expert)

        y_preds = y_preds.view(-1, self.n_tasks)
        y_batch = y_batch.view(-1, self.n_tasks)
        c_preds = c_preds.view(-1, self.n_concepts)
        c_batch = c_batch.view(-1, self.n_concepts)

        acc_test_task = self.task_accuracy_val(y_preds, y_batch)
        acc_test_concepts = self.concept_accuracy_val(c_preds, c_batch)

        # Coverage: share of predictions the model makes itself (argmax != defer slot).
        # A CE-nohuman group never trains its defer logit, so it never defers.
        if self.psi_loss_concepts == "CE-nohuman":
            cov_conc = torch.ones((), device=c_logit.device)
        else:
            cov_conc = (c_logit.argmax(-1) != 2).float().mean()
        if self.psi_loss_tasks == "CE-nohuman":
            cov_task = torch.ones((), device=y_logit_pipe.device)
        else:
            cov_task = (y_logit_pipe.argmax(-1) != self.n_classes).float().mean()

        self.log("cov_conc_val", cov_conc, batch_size=x_batch.shape[0])
        self.log("cov_task_val", cov_task, batch_size=x_batch.shape[0])

        err_rate_task = (y_preds != y_batch).float().mean()
        err_rate_concepts = (c_preds != c_batch).float().mean()

        self.err_c_y_val.append(.5 * (err_rate_task + err_rate_concepts))
        self.err_rate_task_val.append(err_rate_task)

        return loss
    
    def on_validation_epoch_end(self):
        avg_error_rate = torch.stack([x for x in self.err_c_y_val]).mean()
        avg_err_task = torch.stack([x for x in self.err_rate_task_val]).mean()

        self.log("err_c_y_val", avg_error_rate, prog_bar=True)
        self.log("err_task_val", avg_err_task, prog_bar=True)

        self.err_rate_task_val.clear()
        self.err_c_y_val.clear()

    def predict_step(self, batch: torch.Tensor):
        y_preds, c_preds, c_logit, y_logit, _, _, _, _ = self._run_step(batch)
        return {
            "y_preds": y_preds,
            "c_preds": c_preds,
            "c_logit": c_logit,
            "y_logit": y_logit,
        }