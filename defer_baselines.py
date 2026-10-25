"""
Implements the consistent losses as described in the paper
"Mitigating Underfitting in Learning to Defer with Consistent Losses"
by Liu, Cao, Zhang et al. (2024).
"""
from typing import Any

import pytorch_lightning as pl
import torch
from torch.optim import Adam
import torchmetrics
from scc.networks import WideResNet, InceptionV1, InceptionV3, Resn34, CHDenseNet
import torchvision.models as models

DICT_MODEL_PATHS = {"cifar10h": "models/cifar10h/cifar10h_wrn28_4_GC_200epochs.pt",
                    "CUB": "models/cub/ResNet34_100epochs_42.pt"}


class MLPConsistent(pl.LightningModule):
    """
    A consistent defer classifier is a multi-class
    classifier that has an additional label for each
    of the available experts. The loss function is
    then consistent with the deferment of the expert.
    """

    def __init__(
        self,
        input_dim: int,
        emb_size: int,
        n_classes: int,
        n_tasks: int,
        n_experts: int,
        psi_loss: str = "CE",
        label_smoothing: bool = False,
        cost: float = 0.5,
        learning_rate: float = 1e-3,
        task_lambdas: list = None,
        epsilon=1e-5,
        alpha=0,
    ):
        super(MLPConsistent, self).__init__()

        # Store internal parameters
        self.n_classes = n_classes
        self.n_experts = n_experts
        self.psi_loss = psi_loss
        self.label_smoothing_free = label_smoothing
        self.learning_rate = learning_rate
        self.cost = cost
        self.n_tasks = n_tasks
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.eps = epsilon
        self.alpha = alpha
        # Check if psi loss is valid
        if psi_loss not in ["CE", "OVA", "ASM", "RS", "CE-nohuman"]:
            raise ValueError(f"Invalid psi loss function {psi_loss}.")

        # Check if the number of experts is valid
        if self.n_experts not in [0, 1, 2]:
            raise NotImplementedError("Only one or zero experts are supported.")

        # Check that the cost is valid
        assert self.cost <= 1 and self.cost >= 0

        if task_lambdas is None:
            tmp = [1 for _ in range(n_tasks)]
            self.task_lambdas = torch.ones(
                size=(1, n_tasks, n_experts), device=self.device
            )
            self.task_lambdas[0, :, 0] = torch.tensor(tmp, device=self.device)
        else:
            self.task_lambdas = torch.ones(
                size=(1, n_tasks, n_experts), device=self.device
            )
            self.task_lambdas[0, :, 0] = torch.tensor(task_lambdas, device=self.device)
        # Initialize the model
        self.n_outputs_tasks = (n_classes + n_experts) * n_tasks
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, self.n_outputs_tasks),
        )

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _psi(self, y_logit: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """
        Computes the learning to defer function, a.k.a., the psi function.
        The tensor y_pred contains the logits for the predicted labels and
        has shape (batch_size, n_classes + n_experts). The tensor y_batch
        has shape (batch_size, 1) and contains the target labels. The function
        returns a tensor of shape (batch_size, 1).
        """
        # check datatype
        assert y_target.dtype == torch.int64

        if self.psi_loss == "CE":
            # Compute the cross-entropy loss
            log_probs = torch.log_softmax(y_logit, dim=1)
            true_log_probs = torch.gather(log_probs, 1, y_target).squeeze(1)
            return -true_log_probs
        else:
            raise NotImplementedError("Only CE is supported.")

    def _vector_loss(
        self,
        y_logit: torch.Tensor,
        y_batch: torch.Tensor,
        y_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the loss function.

        Parameters
        ----------
        y_logit : torch.Tensor
            The logits of the network (batch_size, n_classes+n_experts).
        y_batch : torch.Tensor
            The target labels (batch_size, 1).
        y_expert : torch.Tensor
            The expert labels (batch_size, n_experts).
        """
        assert len(y_batch.size()) == 3
        assert len(y_expert.size()) == 3
        assert len(y_logit.size()) == 3
        self.indicator = self.n_classes
        lambdas = self.task_lambdas.to(self.device)  # 1 x n_tasks x 1
        mask_right = y_expert == y_batch
        # mask_right = mask_right.squeeze(1)
        # double check when increasing the
        mask_wrong = y_expert != y_batch
        # mask_wrong = mask_wrong.squeeze(1)
        if self.psi_loss == "CE":
            # first term of the loss by Liu et al. 2024 (no label smoothing)
            l_1_all = -torch.log_softmax(y_logit, dim=-1)
            l_1 = torch.gather(l_1_all, -1, y_batch)
            # second term for the loss function defined by Liu (still label smoothing)
            if self.label_smoothing_free:
                l_2 = (
                    +(self.cost)
                    * (
                        torch.min(l_1_all[:, :, : self.indicator], dim=-1)
                    ).values.unsqueeze(-1)
                    * mask_wrong.to(torch.float32)
                )
            else:
                l_2 = (
                    +(self.cost)
                    * (torch.sum(l_1_all[:, :, : self.indicator], dim=-1)).unsqueeze(-1)
                    * mask_wrong.to(torch.float32)
                )
            # third term for the loss function defined by Liu (no label smoothing)
            l_3 = (
                +(1 - self.cost)
                * mask_right.to(torch.float32)
                * l_1_all[:, :, self.indicator :]
            )
            loss = (
                torch.mean(l_1 * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3 * lambdas)
            )
            # self.log("loss", loss)
        elif self.psi_loss == "RS":
            exp_1_all = torch.exp(y_logit)
            exp_gt = torch.gather(exp_1_all, -1, y_batch)
            exp_humans = exp_1_all[:, :, self.indicator :]
            loss_vector = -2 * torch.log(
                (exp_gt + exp_humans * mask_right.to(torch.float32))
                / torch.sum(exp_1_all, dim=-1).unsqueeze(-1)
                + self.eps
            )
            loss = (1 - self.alpha) * torch.mean(loss_vector)
            if self.alpha > 0:
                l_1_all = -torch.log_softmax(y_logit[:, :, : self.indicator], dim=-1)
                l_1 = torch.gather(l_1_all, -1, y_batch)
                loss += (self.alpha) * torch.mean(l_1 * lambdas)
            if self.label_smoothing_free:
                raise NotImplementedError(
                    "Label smoothing is not implemented for RS loss."
                )  # TODO
        elif self.psi_loss == "OVA":
            l_1_all_neg = -torch.nn.functional.logsigmoid(
                y_logit
            )  # log(1-exp(-g(x))) on all ys
            l_1_all_pos = -torch.nn.functional.logsigmoid(
                -y_logit
            )  # log(1-exp(g(x))) on all ys
            l_1_part1 = torch.gather(
                l_1_all_neg, -1, y_batch
            )  # log(1-exp(-g(x))) on ground truth
            l_1_part2 = torch.gather(
                l_1_all_pos, -1, y_batch
            )  # log(1-exp(g(x))) on ground truth
            l_1 = (
                l_1_part1 + torch.sum(l_1_all_pos, dim=-1).unsqueeze(-1) - l_1_part2
            )  # log(1-exp(-g(x))) on ground truth + sum(log(1-exp(g(x))) on all ys) - log(1-exp(g(x))) on ground truth
            if self.label_smoothing_free:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        torch.min(
                            l_1_all_neg[:, :, : self.indicator]
                            + torch.sum(l_1_all_pos, dim=-1).unsqueeze(-1)
                            - l_1_all_pos[:, :, : self.indicator],
                            dim=-1,
                        ).values.unsqueeze(
                            -1
                        )  # log(1-exp(-g(x))) on all ys + sum(log(1-exp(g(x))) on all ys) - log(1-exp(g(x))) on al
                    )
                )
            else:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        (
                            torch.sum(
                                l_1_all_neg[:, :, : self.indicator], dim=-1
                            )  # log(1-exp(-g(x))) on all ys
                            + torch.sum(l_1_all_pos[:, :, : self.indicator], dim=-1)
                        )
                        * (
                            self.indicator - self.n_experts
                        )  # log(1-exp(g(x))) on all ys * (n_classes-n_experts)
                        + torch.sum(l_1_all_pos[:, :, self.indicator :], dim=-1)
                        * (self.n_experts)  # log(1-exp(-g(x))) on all n_experts
                    ).unsqueeze(-1)
                )
            l_3 = (
                +mask_right.to(torch.float32)
                * (1 - self.cost)
                * (
                    torch.sum(
                        l_1_all_neg[:, :, self.indicator :]
                        - l_1_all_pos[:, :, self.indicator :],
                        dim=-1,
                    ).unsqueeze(-1)
                )
            )
            loss = (
                torch.mean(l_1 * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3 * lambdas)
            )
        elif self.psi_loss == "ASM":
            logsm_onlyML = -torch.nn.functional.log_softmax(
                y_logit[:, :, : self.indicator], dim=-1
            )  # log softmax only on ML classes
            sm_onlyML_gt = torch.gather(
                logsm_onlyML, -1, y_batch
            )  # ground truth log softmax value
            exp_onlyML = torch.exp(
                y_logit[:, :, : self.indicator]
            )  # exponential values only on ML classes
            exp_all = torch.exp(y_logit)  # exponential values on all logits
            sum_exp_onlyML = torch.sum(exp_onlyML, dim=-1).unsqueeze(
                -1
            )  # sum of exponential values only on ML classes
            sum_exp_all = torch.sum(exp_all, dim=-1).unsqueeze(-1)
            max_exp_onlyML = torch.max(exp_onlyML, dim=-1).values.unsqueeze(
                -1
            )  # max of exponential values only on ML classes
            numerator_1 = (
                sum_exp_onlyML - max_exp_onlyML
            )  # numerator of the second part (ASM) in Cao et al., 2024
            denominator_1 = (
                sum_exp_all - max_exp_onlyML + self.eps
            )  # denominator of the second part (ASM) in Cao et al., 2024
            exp_expert = exp_all[:, :, self.indicator :]
            l_1 = sm_onlyML_gt - torch.log(
                (numerator_1 / denominator_1) + self.eps
            )  # first part of the loss function in Cao et al., 2024
            sum_all_logsm_onlyML = torch.sum(logsm_onlyML, dim=-1).unsqueeze(
                -1
            )  # sum of log softmax values only on ML classes
            sum_all_part2 = self.indicator * torch.log(
                numerator_1 / denominator_1 + self.eps
            )  # sum of log softmax values on all classes
            if self.label_smoothing_free:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        torch.min(
                            logsm_onlyML
                            - torch.log(1 - (exp_expert / denominator_1) + self.eps),
                            dim=-1,
                        ).values.unsqueeze(-1)
                        # second part of the loss function in Cao et al., 2024
                    )
                )
            else:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (sum_all_logsm_onlyML - sum_all_part2)
                )  # second part of the loss function in Cao et al., 2024

            l_3 = (
                +mask_right.to(torch.float32)
                * (1 - self.cost)
                * (
                    -torch.log(exp_expert / denominator_1 + self.eps)
                    + torch.log((numerator_1 / denominator_1) + self.eps)
                )
            )
            # third part of the loss function in Cao et al., 2024
            loss = (
                torch.mean(l_1 * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3 * lambdas)
            )
        elif self.psi_loss == "CE-nohuman":
            l_1_all = -torch.log_softmax(y_logit[:, :, : self.indicator], dim=-1)
            l_1 = torch.gather(l_1_all, -1, y_batch)
            loss = torch.mean(l_1 * lambdas)
        self.log("loss", loss)
        return loss

    def forward(self, x_batch: torch.Tensor):
        return self.model.forward(x_batch)

    def predict_from_logits(
        self, y_logit: torch.Tensor, y_expert: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Compute the predictions
        preds = torch.argmax(y_logit, dim=-1).unsqueeze(1)
        if self.n_experts > 0:
            assert self.n_experts == 1
            assert y_expert is not None
            # Mask for index of expert
            expert_id = self.n_classes
            mask_expert = preds == expert_id
            # Replace the expert prediction with the expert label
            preds = torch.where(mask_expert, y_expert, preds)
        return preds

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # type: ignore
        """Computes the training loss."""

        if self.n_experts == 0:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = None
        else:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = batch[2]

        # Compute Logits
        y_logit = self.model.forward(x_batch)
        y_logit = y_logit.view(-1, self.n_tasks, self.n_classes + self.n_experts)
        # Compute the loss
        # loss_val = self._loss(y_logit, y_batch, y_expert=y_expert)
        if self.n_tasks == 1:
            y_batch = y_batch.unsqueeze(-1)
        if self.n_experts > 0:
            loss_val = 0
            y_expert = y_expert.view(-1, self.n_tasks, self.n_experts)
            loss_val = self._vector_loss(y_logit, y_batch, y_expert)
        self.log("loss", loss_val)
        if self.n_experts == 0:
            preds = self.predict_from_logits(y_logit, y_expert=None)
        else:
            preds = self.predict_from_logits(y_logit, y_expert=y_expert)
        acc_val = self.accuracy(preds, y_batch)
        self.log("train_acc", acc_val)

        return loss_val

    def on_train_epoch_end(self) -> None:
        """Log accuracy."""
        self.log("train_acc_epoch", self.accuracy, prog_bar=True)


class CBMConsistent(pl.LightningModule):
    """
    A consistent concept bottleneck model implementation for synth data
    """

    def __init__(
        self,
        input_dim: int,
        emb_size: int,
        n_classes: int,
        n_concepts: int,
        n_tasks: int,
        n_experts: int,
        psi_loss: str = "CE",
        label_smoothing: bool = False,
        cost: float = 0.5,
        learning_rate: float = 1e-3,
        training_mode: str = "independent",
        concept_lambdas: list = None,
        task_lambdas: list = None,
        epsilon: float = 1e-8,
        alpha: float = 0,
        psi_loss_concepts=None,
        psi_loss_tasks=None,
    ):
        super(CBMConsistent, self).__init__()

        # Store internal parameters
        self.n_concepts = n_concepts  # for now all binary concepts
        self.n_classes = n_classes  # for now the same for all final_tasks
        self.n_tasks = n_tasks  # different tasks possible
        self.n_experts = (
            n_experts  # for now same number of experts for each concept/task
        )
        self.psi_loss = psi_loss
        if psi_loss_tasks is None:
            self.psi_loss_tasks = psi_loss
        else:
            self.psi_loss_tasks = psi_loss_tasks
        if psi_loss_concepts is None:
            self.psi_loss_concepts = psi_loss
        else:
            self.psi_loss_concepts = psi_loss_concepts
        self.label_smoothing_free = label_smoothing
        self.learning_rate = learning_rate
        self.cost = cost  # to decide how to implement it for multi concpets/tasks and multi experts
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.eps = epsilon
        self.alpha = alpha
        if concept_lambdas is None:
            tmp = [1 for _ in range(n_concepts)]
            self.concept_lambdas = torch.ones(
                size=(1, n_concepts, 1), device=self.device
            )
            self.concept_lambdas[0, :, 0] = torch.tensor(tmp, device=self.device)
        else:
            self.concept_lambdas = torch.ones(
                size=(1, n_concepts, 1), device=self.device
            )
            self.concept_lambdas[0, :, 0] = torch.tensor(
                concept_lambdas, device=self.device
            )
        if task_lambdas is None:
            tmp = [1 for _ in range(n_tasks)]
            self.task_lambdas = torch.ones(size=(1, n_tasks, 1), device=self.device)
            self.task_lambdas[0, :, 0] = torch.tensor(tmp, device=self.device)
        else:
            self.task_lambdas = torch.ones(size=(1, n_tasks, 1), device=self.device)
            self.task_lambdas[0, :, 0] = torch.tensor(task_lambdas, device=self.device)
        # Check if psi loss is valid
        if self.psi_loss not in ["CE", "OVA", "ASM", "RS", "CE-nohuman"]:
            raise ValueError(f"Invalid psi loss function {psi_loss} not in CE.")
        if self.psi_loss_tasks not in ["CE", "OVA", "ASM", "RS", "CE-nohuman"]:
            raise ValueError(
                f"Invalid psi loss function {self.psi_loss_tasks} not in CE."
            )
        if self.psi_loss_concepts not in ["CE", "OVA", "ASM", "RS", "CE-nohuman"]:
            raise ValueError(
                f"Invalid psi loss function {self.psi_loss_concepts} not in CE."
            )
        # Check if the number of experts is valid
        if self.n_experts not in [0, 1]:
            raise NotImplementedError("Only one or zero experts are supported.")

        # Check that the cost is valid
        assert self.cost <= 1 and self.cost >= 0
        self.epsilon = 1e-8
        # Initialize the model
        n_outputs = n_classes + n_experts
        self.n_outputs_concepts = (
            2 + n_experts
        ) * n_concepts  # NOTE: this is for binary concepts
        self.n_outputs_tasks = (n_classes + n_experts) * n_tasks
        # TODO: take model_concept_base and model_base as input rather than using the linear one
        self.model_concept = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(input_dim, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, 2 + n_experts),
                )
                for _ in range(n_concepts)
            ]
        )
        self.model = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(n_concepts, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, n_classes + n_experts),
                )
                for _ in range(n_tasks)
            ]
        )
        self.training_mode = training_mode

    def configure_optimizers(self):
        # NOTE: might be a concern?
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def concept_forward(
        self, x_batch: torch.Tensor
    ):  # something smarter can be done here
        c_logit = []
        for module in self.model_concept:
            c_logit.append(module.forward(x_batch))
        return torch.stack(c_logit, dim=1)

    def task_forward(self, c_batch: torch.Tensor):  # something smarter can be done here
        y_logit = []
        for module in self.model:
            y_logit.append(module.forward(c_batch))
        return torch.stack(y_logit, dim=1)

    def _vector_loss(
        self,
        y_logit: torch.Tensor,
        y_batch: torch.Tensor,
        y_expert: torch.Tensor | None = None,
        concepts: bool = False,
    ) -> torch.Tensor:
        """
        Computes the loss function.

        Parameters
        ----------
        y_logit : torch.Tensor
            The logits of the network (batch_size, n_classes+n_experts).
        y_batch : torch.Tensor
            The target labels (batch_size, 1).
        y_expert : torch.Tensor
            The expert labels (batch_size, n_experts).
        """
        assert len(y_batch.size()) == 3
        assert len(y_expert.size()) == 3
        assert len(y_logit.size()) == 3
        if concepts:
            self.indicator = 2
            lambdas = self.concept_lambdas.to(self.device)
            self.psi_loss = self.psi_loss_concepts
        else:
            self.indicator = self.n_classes
            lambdas = self.task_lambdas.to(self.device)
            self.psi_loss = self.psi_loss_tasks
        mask_right = y_expert == y_batch
        # mask_right = mask_right.squeeze(1)
        # double check when increasing the
        mask_wrong = y_expert != y_batch
        # mask_wrong = mask_wrong.squeeze(1)
        if self.psi_loss == "CE":
            # first term of the loss by Liu et al. 2024 (no label smoothing)
            l_1_all = -torch.log_softmax(y_logit, dim=-1)
            l_1 = torch.gather(l_1_all, -1, y_batch)
            # second term for the loss function defined by Liu (still label smoothing)
            if self.label_smoothing_free:
                l_2 = (
                    +(self.cost)
                    * (
                        torch.min(l_1_all[:, :, : self.indicator], dim=-1)
                    ).values.unsqueeze(-1)
                    * mask_wrong.to(torch.float32)
                )
            else:
                l_2 = (
                    +(self.cost)
                    * (torch.sum(l_1_all[:, :, : self.indicator], dim=-1)).unsqueeze(-1)
                    * mask_wrong.to(torch.float32)
                )
            # third term for the loss function defined by Liu (no label smoothing)
            l_3 = (
                +(1 - self.cost)
                * mask_right.to(torch.float32)
                * l_1_all[:, :, self.indicator :]
            )
            loss = (
                torch.mean(l_1 * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3 * lambdas)
            )
            # self.log("loss", loss)
        elif self.psi_loss == "RS":
            exp_1_all = torch.exp(y_logit)
            exp_gt = torch.gather(exp_1_all, -1, y_batch)
            exp_humans = exp_1_all[:, :, self.indicator :]
            loss_vector = -2 * torch.log(
                (exp_gt + exp_humans * mask_right.to(torch.float32))
                / torch.sum(exp_1_all, dim=-1).unsqueeze(-1)
                + self.eps
            )
            loss = (1 - self.alpha) * torch.mean(loss_vector)
            if self.alpha > 0:
                l_1_all = -torch.log_softmax(y_logit[:, :, : self.indicator], dim=-1)
                l_1 = torch.gather(l_1_all, -1, y_batch)
                loss += (self.alpha) * torch.mean(l_1 * lambdas)
            if self.label_smoothing_free:
                raise NotImplementedError(
                    "Label smoothing is not implemented for RS loss."
                )  # TODO
        elif self.psi_loss == "OVA":
            l_1_all_neg = -torch.nn.functional.logsigmoid(
                y_logit
            )  # log(1-exp(-g(x))) on all ys
            l_1_all_pos = -torch.nn.functional.logsigmoid(
                -y_logit
            )  # log(1-exp(g(x))) on all ys
            l_1_part1 = torch.gather(
                l_1_all_neg, -1, y_batch
            )  # log(1-exp(-g(x))) on ground truth
            l_1_part2 = torch.gather(
                l_1_all_pos, -1, y_batch
            )  # log(1-exp(g(x))) on ground truth
            l_1 = (
                l_1_part1 + torch.sum(l_1_all_pos, dim=-1).unsqueeze(-1) - l_1_part2
            )  # log(1-exp(-g(x))) on ground truth + sum(log(1-exp(g(x))) on all ys) - log(1-exp(g(x))) on ground truth
            if self.label_smoothing_free:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        torch.min(
                            l_1_all_neg[:, :, : self.indicator]
                            + torch.sum(l_1_all_pos, dim=-1).unsqueeze(-1)
                            - l_1_all_pos[:, :, : self.indicator],
                            dim=-1,
                        ).values.unsqueeze(
                            -1
                        )  # log(1-exp(-g(x))) on all ys + sum(log(1-exp(g(x))) on all ys) - log(1-exp(g(x))) on al
                    )
                )
            else:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        (
                            torch.sum(
                                l_1_all_neg[:, :, : self.indicator], dim=-1
                            )  # log(1-exp(-g(x))) on all ys
                            + torch.sum(l_1_all_pos[:, :, : self.indicator], dim=-1)
                        )
                        * (
                            self.indicator - self.n_experts
                        )  # log(1-exp(g(x))) on all ys * (n_classes-n_experts)
                        + torch.sum(l_1_all_pos[:, :, self.indicator :], dim=-1)
                        * (self.n_experts)  # log(1-exp(-g(x))) on all n_experts
                    ).unsqueeze(-1)
                )
            l_3 = (
                +mask_right.to(torch.float32)
                * (1 - self.cost)
                * (
                    torch.sum(
                        l_1_all_neg[:, :, self.indicator :]
                        - l_1_all_pos[:, :, self.indicator :],
                        dim=-1,
                    ).unsqueeze(-1)
                )
            )
            loss = (
                torch.mean(l_1 * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3 * lambdas)
            )
        elif self.psi_loss == "ASM":
            logsm_onlyML = -torch.nn.functional.log_softmax(
                y_logit[:, :, : self.indicator], dim=-1
            )  # log softmax only on ML classes
            sm_onlyML_gt = torch.gather(
                logsm_onlyML, -1, y_batch
            )  # ground truth log softmax value
            exp_onlyML = torch.exp(
                y_logit[:, :, : self.indicator]
            )  # exponential values only on ML classes
            exp_all = torch.exp(y_logit)  # exponential values on all logits
            sum_exp_onlyML = torch.sum(exp_onlyML, dim=-1).unsqueeze(
                -1
            )  # sum of exponential values only on ML classes
            sum_exp_all = torch.sum(exp_all, dim=-1).unsqueeze(-1)
            max_exp_onlyML = torch.max(exp_onlyML, dim=-1).values.unsqueeze(
                -1
            )  # max of exponential values only on ML classes
            numerator_1 = (
                sum_exp_onlyML - max_exp_onlyML
            )  # numerator of the second part (ASM) in Cao et al., 2024
            denominator_1 = (
                sum_exp_all - max_exp_onlyML + self.eps
            )  # denominator of the second part (ASM) in Cao et al., 2024
            exp_expert = exp_all[:, :, self.indicator :]
            l_1 = sm_onlyML_gt - torch.log(
                (numerator_1 / denominator_1) + self.eps
            )  # first part of the loss function in Cao et al., 2024
            sum_all_logsm_onlyML = torch.sum(logsm_onlyML, dim=-1).unsqueeze(
                -1
            )  # sum of log softmax values only on ML classes
            sum_all_part2 = self.indicator * torch.log(
                numerator_1 / denominator_1 + self.eps
            )  # sum of log softmax values on all classes
            if self.label_smoothing_free:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        torch.min(
                            logsm_onlyML
                            - torch.log(1 - (exp_expert / denominator_1) + self.eps),
                            dim=-1,
                        ).values.unsqueeze(-1)
                        # second part of the loss function in Cao et al., 2024
                    )
                )
            else:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (sum_all_logsm_onlyML - sum_all_part2)
                )  # second part of the loss function in Cao et al., 2024

            l_3 = (
                +mask_right.to(torch.float32)
                * (1 - self.cost)
                * (
                    -torch.log(exp_expert / denominator_1 + self.eps)
                    + torch.log((numerator_1 / denominator_1) + self.eps)
                )
            )
            # third part of the loss function in Cao et al., 2024
            loss = (
                torch.mean(l_1 * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3 * lambdas)
            )
        elif self.psi_loss == "CE-nohuman":
            l_1_all = -torch.log_softmax(y_logit[:, :, : self.indicator], dim=-1)
            l_1 = torch.gather(l_1_all, -1, y_batch)
            loss = torch.mean(l_1 * lambdas)
        self.log("loss", loss)
        return loss

    def forward(
        self, x_batch: torch.Tensor, y_expert: torch.Tensor | None = None
    ) -> torch.Tensor:
        if (y_expert is None) or (self.psi_loss == "CE-nohuman"):
            return self.concept_forward(x_batch)
        else:
            c_logit = self.concept_forward(x_batch)
            c_logit = c_logit.view(-1, self.n_concepts, 2 + self.n_experts)
            preds = torch.argmax(c_logit, dim=-1).unsqueeze(-1)
            if self.n_experts > 0:
                assert self.n_experts == 1
                assert y_expert is not None
                # Mask for index of expert
                expert_id = self.n_classes
                mask_expert = preds == expert_id
                # Replace the expert prediction with the expert label
                preds = torch.where(mask_expert, y_expert, preds)
            return self.task_forward(preds)

    def predict_from_logits(
        self, y_logit: torch.Tensor, y_expert: torch.Tensor | None = None, concepts=False
    ) -> torch.Tensor:
        if concepts is True:
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

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # type: ignore
        """Computes the training loss."""

        if self.n_experts == 0:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = None
            c_batch = batch[3]
            c_expert = None
        else:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = batch[2]
            c_batch = batch[3]
            c_expert = batch[4]
        if self.n_tasks == 1:
            y_batch = y_batch.unsqueeze(-1)
        c_batch = c_batch.unsqueeze(-1)
        # Compute Logits (for concepts)
        c_logit = self.concept_forward(x_batch)
        c_logit = c_logit.view(-1, self.n_concepts, 2 + self.n_experts)
        # Compute Logits (for the final task)
        if self.training_mode == "independent":
            y_logit = self.task_forward(c_batch.float().squeeze(-1))
        elif self.training_mode == "sequential":
            c_preds = self.predict_from_logits(c_logit, y_expert=c_expert)
            y_logit = self.task_forward(c_preds.float())
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
            loss_con = self._vector_loss(c_logit, c_batch, c_expert, concepts=True)
            loss_task = self._vector_loss(y_logit, y_batch, y_expert, concepts=False)
            loss_val = loss_con + loss_task
        else:
            raise NotImplementedError(
                "Only one expert is supported. If you want the baseline no expert use psi_loss == 'CE-nohuman'"
            )
        self.log("loss", loss_val)
        # Compute the predictions
        preds = self.predict_from_logits(y_logit, y_expert)
        c_preds = self.predict_from_logits(c_logit, c_expert)
        # Compute and log the accuracy
        acc_val = self.accuracy(preds, y_batch)
        self.log("train_acc", acc_val)
        acc_c_val = self.accuracy(c_preds, c_batch)
        self.log("train_c_acc", acc_c_val)
        return loss_val



class CBMConsistentVariant(pl.LightningModule):
    """
    TODO: finish the implementation of the CBMConsistent model.
    """

    def __init__(
        self,
        input_dim: int,
        emb_size: int,
        n_classes: int,
        n_concepts: int,
        n_tasks: int,
        n_experts: int,
        psi_loss: str = "CE",
        label_smoothing: bool = False,
        cost: float = 0.5,
        learning_rate: float = 1e-3,
        training_mode: str = "independent",
        concept_lambdas: list = None,
        task_lambdas: list = None,
        epsilon: float = 1e-8,
        alpha: float = 0,
        psi_loss_concepts=None,
        psi_loss_tasks=None,
    ):
        super(CBMConsistentVariant, self).__init__()

        # Store internal parameters
        self.n_concepts = n_concepts  # for now all binary concepts
        self.n_classes = n_classes  # for now the same for all final_tasks
        self.n_tasks = n_tasks  # different tasks possible
        self.n_experts = (
            n_experts  # for now same number of experts for each concept/task
        )
        self.psi_loss = psi_loss
        if psi_loss_tasks is None:
            self.psi_loss_tasks = psi_loss
        else:
            self.psi_loss_tasks = psi_loss_tasks
        if psi_loss_concepts is None:
            self.psi_loss_concepts = psi_loss
        else:
            self.psi_loss_concepts = psi_loss_concepts
        self.label_smoothing_free = label_smoothing
        self.learning_rate = learning_rate
        self.cost = cost  # to decide how to implement it for multi concpets/tasks and multi experts
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.eps = epsilon
        self.alpha = alpha
        if concept_lambdas is None:
            tmp = [1 for _ in range(n_concepts)]
            self.concept_lambdas = torch.ones(
                size=(1, n_concepts, 1), device=self.device
            )
            self.concept_lambdas[0, :, 0] = torch.tensor(tmp, device=self.device)
        else:
            self.concept_lambdas = torch.ones(
                size=(1, n_concepts, 1), device=self.device
            )
            self.concept_lambdas[0, :, 0] = torch.tensor(
                concept_lambdas, device=self.device
            )
        if task_lambdas is None:
            tmp = [1 for _ in range(n_tasks)]
            self.task_lambdas = torch.ones(size=(1, n_tasks, 1), device=self.device)
            self.task_lambdas[0, :, 0] = torch.tensor(tmp, device=self.device)
        else:
            self.task_lambdas = torch.ones(size=(1, n_tasks, 1), device=self.device)
            self.task_lambdas[0, :, 0] = torch.tensor(task_lambdas, device=self.device)
        # Check if psi loss is valid
        if self.psi_loss not in ["CE", "OVA", "ASM", "RS", "CE-nohuman"]:
            raise ValueError(f"Invalid psi loss function {psi_loss} not in CE.")
        if self.psi_loss_tasks not in ["CE", "OVA", "ASM", "RS", "CE-nohuman"]:
            raise ValueError(
                f"Invalid psi loss function {self.psi_loss_tasks} not in CE."
            )
        if self.psi_loss_concepts not in ["CE", "OVA", "ASM", "RS", "CE-nohuman"]:
            raise ValueError(
                f"Invalid psi loss function {self.psi_loss_concepts} not in CE."
            )
        # Check if the number of experts is valid
        if self.n_experts not in [0, 1]:
            raise NotImplementedError("Only one or zero experts are supported.")

        # Check that the cost is valid
        assert self.cost <= 1 and self.cost >= 0
        self.epsilon = 1e-8
        # Initialize the model
        n_outputs = n_classes + n_experts
        self.n_outputs_concepts = (
            2 + n_experts
        ) * n_concepts  # NOTE: this is for binary concepts
        self.n_outputs_tasks = (n_classes + n_experts) * n_tasks
        # TODO: take model_concept_base and model_base as input rather than using the linear one
        self.shared_encoder = torch.nn.Sequential(
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, emb_size*2),
                torch.nn.LeakyReLU()
        ))
        self.model_concept = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(emb_size*2, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, 2 + n_experts),
                )
                for _ in range(n_concepts)
            ]
        )
        self.model = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(n_concepts, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, n_classes + n_experts),
                )
                for _ in range(n_tasks)
            ]
        )
        self.training_mode = training_mode

    def configure_optimizers(self):
        # NOTE: might be a concern?
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def concept_forward(
        self, x_batch: torch.Tensor
    ):  # something smarter can be done here
        c_logit = []
        x = self.shared_encoder(x_batch)
        for module in self.model_concept:
            c_logit.append(module.forward(x))
        return torch.stack(c_logit, dim=1)

    def task_forward(self, c_batch: torch.Tensor):  # something smarter can be done here
        y_logit = []
        for module in self.model:
            y_logit.append(module.forward(c_batch))
        return torch.stack(y_logit, dim=1)

    def _vector_loss(
        self,
        y_logit: torch.Tensor,
        y_batch: torch.Tensor,
        y_expert: torch.Tensor | None = None,
        concepts: bool = False,
    ) -> torch.Tensor:
        """
        Computes the loss function.

        Parameters
        ----------
        y_logit : torch.Tensor
            The logits of the network (batch_size, n_classes+n_experts).
        y_batch : torch.Tensor
            The target labels (batch_size, 1).
        y_expert : torch.Tensor
            The expert labels (batch_size, n_experts).
        """
        assert len(y_batch.size()) == 3
        assert len(y_expert.size()) == 3
        assert len(y_logit.size()) == 3
        if concepts:
            self.indicator = 2
            lambdas = self.concept_lambdas.to(self.device)
            self.psi_loss = self.psi_loss_concepts
        else:
            self.indicator = self.n_classes
            lambdas = self.task_lambdas.to(self.device)
            self.psi_loss = self.psi_loss_tasks
        mask_right = y_expert == y_batch
        # mask_right = mask_right.squeeze(1)
        # double check when increasing the
        mask_wrong = y_expert != y_batch
        # mask_wrong = mask_wrong.squeeze(1)
        if self.psi_loss == "CE":
            # first term of the loss by Liu et al. 2024 (no label smoothing)
            l_1_all = -torch.log_softmax(y_logit, dim=-1)
            l_1 = torch.gather(l_1_all, -1, y_batch)
            # second term for the loss function defined by Liu (still label smoothing)
            if self.label_smoothing_free:
                l_2 = (
                    +(self.cost)
                    * (
                        torch.min(l_1_all[:, :, : self.indicator], dim=-1)
                    ).values.unsqueeze(-1)
                    * mask_wrong.to(torch.float32)
                )
            else:
                l_2 = (
                    +(self.cost)
                    * (torch.sum(l_1_all[:, :, : self.indicator], dim=-1)).unsqueeze(-1)
                    * mask_wrong.to(torch.float32)
                )
            # third term for the loss function defined by Liu (no label smoothing)
            l_3 = (
                +(1 - self.cost)
                * mask_right.to(torch.float32)
                * l_1_all[:, :, self.indicator :]
            )
            loss = (
                torch.mean(l_1 * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3 * lambdas)
            )
            # self.log("loss", loss)
        elif self.psi_loss == "RS":
            exp_1_all = torch.exp(y_logit)
            exp_gt = torch.gather(exp_1_all, -1, y_batch)
            exp_humans = exp_1_all[:, :, self.indicator :]
            loss_vector = -2 * torch.log(
                (exp_gt + exp_humans * mask_right.to(torch.float32))
                / torch.sum(exp_1_all, dim=-1).unsqueeze(-1)
                + self.eps
            )
            loss = (1 - self.alpha) * torch.mean(loss_vector)
            if self.alpha > 0:
                l_1_all = -torch.log_softmax(y_logit[:, :, : self.indicator], dim=-1)
                l_1 = torch.gather(l_1_all, -1, y_batch)
                loss += (self.alpha) * torch.mean(l_1 * lambdas)
            if self.label_smoothing_free:
                raise NotImplementedError(
                    "Label smoothing is not implemented for RS loss."
                )  # TODO
        elif self.psi_loss == "OVA":
            l_1_all_neg = -torch.nn.functional.logsigmoid(
                y_logit
            )  # log(1-exp(-g(x))) on all ys
            l_1_all_pos = -torch.nn.functional.logsigmoid(
                -y_logit
            )  # log(1-exp(g(x))) on all ys
            l_1_part1 = torch.gather(
                l_1_all_neg, -1, y_batch
            )  # log(1-exp(-g(x))) on ground truth
            l_1_part2 = torch.gather(
                l_1_all_pos, -1, y_batch
            )  # log(1-exp(g(x))) on ground truth
            l_1 = (
                l_1_part1 + torch.sum(l_1_all_pos, dim=-1).unsqueeze(-1) - l_1_part2
            )  # log(1-exp(-g(x))) on ground truth + sum(log(1-exp(g(x))) on all ys) - log(1-exp(g(x))) on ground truth
            if self.label_smoothing_free:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        torch.min(
                            l_1_all_neg[:, :, : self.indicator]
                            + torch.sum(l_1_all_pos, dim=-1).unsqueeze(-1)
                            - l_1_all_pos[:, :, : self.indicator],
                            dim=-1,
                        ).values.unsqueeze(
                            -1
                        )  # log(1-exp(-g(x))) on all ys + sum(log(1-exp(g(x))) on all ys) - log(1-exp(g(x))) on al
                    )
                )
            else:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        (
                            torch.sum(
                                l_1_all_neg[:, :, : self.indicator], dim=-1
                            )  # log(1-exp(-g(x))) on all ys
                            + torch.sum(l_1_all_pos[:, :, : self.indicator], dim=-1)
                        )
                        * (
                            self.indicator - self.n_experts
                        )  # log(1-exp(g(x))) on all ys * (n_classes-n_experts)
                        + torch.sum(l_1_all_pos[:, :, self.indicator :], dim=-1)
                        * (self.n_experts)  # log(1-exp(-g(x))) on all n_experts
                    ).unsqueeze(-1)
                )
            l_3 = (
                +mask_right.to(torch.float32)
                * (1 - self.cost)
                * (
                    torch.sum(
                        l_1_all_neg[:, :, self.indicator :]
                        - l_1_all_pos[:, :, self.indicator :],
                        dim=-1,
                    ).unsqueeze(-1)
                )
            )
            loss = (
                torch.mean(l_1 * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3 * lambdas)
            )
        elif self.psi_loss == "ASM":
            logsm_onlyML = -torch.nn.functional.log_softmax(
                y_logit[:, :, : self.indicator], dim=-1
            )  # log softmax only on ML classes
            sm_onlyML_gt = torch.gather(
                logsm_onlyML, -1, y_batch
            )  # ground truth log softmax value
            exp_onlyML = torch.exp(
                y_logit[:, :, : self.indicator]
            )  # exponential values only on ML classes
            exp_all = torch.exp(y_logit)  # exponential values on all logits
            sum_exp_onlyML = torch.sum(exp_onlyML, dim=-1).unsqueeze(
                -1
            )  # sum of exponential values only on ML classes
            sum_exp_all = torch.sum(exp_all, dim=-1).unsqueeze(-1)
            max_exp_onlyML = torch.max(exp_onlyML, dim=-1).values.unsqueeze(
                -1
            )  # max of exponential values only on ML classes
            numerator_1 = (
                sum_exp_onlyML - max_exp_onlyML
            )  # numerator of the second part (ASM) in Cao et al., 2024
            denominator_1 = (
                sum_exp_all - max_exp_onlyML + self.eps
            )  # denominator of the second part (ASM) in Cao et al., 2024
            exp_expert = exp_all[:, :, self.indicator :]
            l_1 = sm_onlyML_gt - torch.log(
                (numerator_1 / denominator_1) + self.eps
            )  # first part of the loss function in Cao et al., 2024
            sum_all_logsm_onlyML = torch.sum(logsm_onlyML, dim=-1).unsqueeze(
                -1
            )  # sum of log softmax values only on ML classes
            sum_all_part2 = self.indicator * torch.log(
                numerator_1 / denominator_1 + self.eps
            )  # sum of log softmax values on all classes
            if self.label_smoothing_free:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        torch.min(
                            logsm_onlyML
                            - torch.log(1 - (exp_expert / denominator_1) + self.eps),
                            dim=-1,
                        ).values.unsqueeze(-1)
                        # second part of the loss function in Cao et al., 2024
                    )
                )
            else:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (sum_all_logsm_onlyML - sum_all_part2)
                )  # second part of the loss function in Cao et al., 2024

            l_3 = (
                +mask_right.to(torch.float32)
                * (1 - self.cost)
                * (
                    -torch.log(exp_expert / denominator_1 + self.eps)
                    + torch.log((numerator_1 / denominator_1) + self.eps)
                )
            )
            # third part of the loss function in Cao et al., 2024
            loss = (
                torch.mean(l_1 * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3 * lambdas)
            )
        elif self.psi_loss == "CE-nohuman":
            l_1_all = -torch.log_softmax(y_logit[:, :, : self.indicator], dim=-1)
            l_1 = torch.gather(l_1_all, -1, y_batch)
            loss = torch.mean(l_1 * lambdas)
        self.log("loss", loss)
        return loss

    def forward(
        self, x_batch: torch.Tensor, y_expert: torch.Tensor | None = None
    ) -> torch.Tensor:
        if (y_expert is None) or (self.psi_loss == "CE-nohuman"):
            return self.concept_forward(x_batch)
        else:
            c_logit = self.concept_forward(x_batch)
            c_logit = c_logit.view(-1, self.n_concepts, 2 + self.n_experts)
            preds = torch.argmax(c_logit, dim=-1).unsqueeze(-1)
            if self.n_experts > 0:
                assert self.n_experts == 1
                assert y_expert is not None
                # Mask for index of expert
                expert_id = self.n_classes
                mask_expert = preds == expert_id
                # Replace the expert prediction with the expert label
                preds = torch.where(mask_expert, y_expert, preds)
            return self.task_forward(preds)

    def predict_from_logits(
        self, y_logit: torch.Tensor, y_expert: torch.Tensor | None = None, concepts=False
    ) -> torch.Tensor:
        if concepts is True:
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

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # type: ignore
        """Computes the training loss."""

        if self.n_experts == 0:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = None
            c_batch = batch[3]
            c_expert = None
        else:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = batch[2]
            c_batch = batch[3]
            c_expert = batch[4]
        if self.n_tasks == 1:
            y_batch = y_batch.unsqueeze(-1)
        c_batch = c_batch.unsqueeze(-1)
        # Compute Logits (for concepts)
        c_logit = self.concept_forward(x_batch)
        c_logit = c_logit.view(-1, self.n_concepts, 2 + self.n_experts)
        # Compute Logits (for the final task)
        if self.training_mode == "independent":
            y_logit = self.task_forward(c_batch.float().squeeze(-1))
        elif self.training_mode == "sequential":
            c_preds = self.predict_from_logits(c_logit, y_expert=c_expert)
            y_logit = self.task_forward(c_preds.float())
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
            loss_con = self._vector_loss(c_logit, c_batch, c_expert, concepts=True)
            loss_task = self._vector_loss(y_logit, y_batch, y_expert, concepts=False)
            loss_val = loss_con + loss_task
        else:
            raise NotImplementedError(
                "Only one expert is supported. If you want the baseline no expert use psi_loss == 'CE-nohuman'"
            )
        self.log("loss", loss_val)
        # Compute the predictions
        preds = self.predict_from_logits(y_logit, y_expert)
        c_preds = self.predict_from_logits(c_logit, c_expert)
        # Compute and log the accuracy
        acc_val = self.accuracy(preds, y_batch)
        self.log("train_acc", acc_val)
        acc_c_val = self.accuracy(c_preds, c_batch)
        self.log("train_c_acc", acc_c_val)
        return loss_val

class CBMConsistentVision(pl.LightningModule):
    """
    The CBMConsistent model using Inception as the main model
    """

    def __init__(
        self,
        model_rep: torch.nn.Module,
        emb_size: int,
        n_classes: int,
        n_concepts: int,
        n_tasks: int,
        n_experts: int,
        psi_loss: str = "CE",
        label_smoothing: bool = False,
        cost: float = 0.5,
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
        optimizer_name="adamW",
        halved: bool = False,
        setup_optimizer: str = "cifar10",
        pretrain_freeze: bool = True
    ):
        super(CBMConsistentVision, self).__init__()

        # Store internal parameters
        self.n_concepts = n_concepts  # for now all binary concepts
        self.n_classes = n_classes  # for now the same for all final_tasks
        self.n_tasks = n_tasks  # different tasks possible
        self.n_experts = (
            n_experts  # for now same number of experts for each concept/task
        )
        self.psi_loss = psi_loss
        if psi_loss_tasks is None:
            self.psi_loss_tasks = psi_loss
        else:
            self.psi_loss_tasks = psi_loss_tasks
        if psi_loss_concepts is None:
            self.psi_loss_concepts = psi_loss
        else:
            self.psi_loss_concepts = psi_loss_concepts
        self.label_smoothing_free = label_smoothing
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.cost = cost  # to decide how to implement it for multi concpets/tasks and multi experts
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.concept_accuracy = torchmetrics.classification.Accuracy(
                task="multilabel", num_labels=self.n_concepts)
        self.accuracy_val = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.concept_accuracy_val = torchmetrics.classification.Accuracy(
            task="multilabel", num_labels=self.n_concepts
        )
        self.eps = epsilon
        if self.psi_loss == "ASM":
            if self.eps < 1e-5:
                self.eps = 1e-5
        self.alpha = alpha
        if concept_lambdas is None:
            tmp = [1 for _ in range(n_concepts)]
            self.concept_lambdas = torch.ones(
                size=(1, n_concepts, 1), device=self.device
            )
            self.concept_lambdas[0, :, 0] = torch.tensor(tmp, device=self.device)
        else:
            self.concept_lambdas = torch.ones(
                size=(1, n_concepts, 1), device=self.device
            )
            self.concept_lambdas[0, :, 0] = torch.tensor(
                concept_lambdas, device=self.device
            )
        if task_lambdas is None:
            tmp = [1 for _ in range(n_tasks)]
            self.task_lambdas = torch.ones(size=(1, n_tasks, 1), device=self.device)
            self.task_lambdas[0, :, 0] = torch.tensor(tmp, device=self.device)
        else:
            self.task_lambdas = torch.ones(size=(1, n_tasks, 1), device=self.device)
            self.task_lambdas[0, :, 0] = torch.tensor(task_lambdas, device=self.device)
        # Check if psi loss is valid
        if self.psi_loss not in ["CE", "OVA", "ASM", "ASM", "RS", "CE-nohuman"]:
            raise ValueError(f"Invalid psi loss function {psi_loss} not in CE.")
        if self.psi_loss_tasks not in ["CE", "OVA", "ASM", "ASM", "RS", "CE-nohuman"]:
            raise ValueError(
                f"Invalid psi loss function {self.psi_loss_tasks} not in CE."
            )
        if self.psi_loss_concepts not in ["CE", "OVA", "ASM", "RS", "ASM", "CE-nohuman"]:
            raise ValueError(
                f"Invalid psi loss function {self.psi_loss_concepts} not in CE."
            )
        # Check if the number of experts is valid
        if self.n_experts not in [0, 1]:
            raise NotImplementedError("Only one or zero experts are supported.")
        # Check that the cost is valid
        assert self.cost <= 1 and self.cost >= 0
        self.epsilon = 1e-8
        self.weights_concepts = weights_concepts
        self.weights_tasks = weights_tasks
        # Initialize the model
        n_outputs = n_classes + n_experts
        self.n_outputs_concepts = (
            2 + n_experts
        ) * n_concepts  # NOTE: this is for binary concepts
        self.n_outputs_tasks = (n_classes + n_experts) * n_tasks
        self.model_rep = model_rep
        self.pretrain_freeze = pretrain_freeze
        if self.pretrain_freeze:
            for param in self.model_rep.parameters():
                param.requires_grad = False
        assert hasattr(self.model_rep, "repr")
        if type(model_rep) == WideResNet:
            self.in_features = self.model_rep.fc2.in_features
        elif type(model_rep) == InceptionV3:
            self.in_features = self.model_rep.fc.in_features
        elif type(model_rep) == Resn34:
            self.in_features = self.model_rep.fc.in_features
        elif type(model_rep) == CHDenseNet:
            self.in_features = self.model_rep.classifier.in_features# self.model_rep.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        if halved:
            self.hidden_layer_out = emb_size/2
        else:
            self.hidden_layer_out = emb_size
        self.model_concept = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.in_features, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, int(self.hidden_layer_out)),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(int(self.hidden_layer_out), 2 + n_experts),
                )
                for _ in range(n_concepts)
            ]
        )
        self.model = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(n_concepts, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, int(self.hidden_layer_out)),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(int(self.hidden_layer_out), n_classes + n_experts),
                )
                for _ in range(n_tasks)
            ]
        )
        self.training_mode = training_mode
        self.err_c_y_val = []
        self.err_rate_task_val = []
        self.setup_optimizer = setup_optimizer

    def repr(self, x_batch: torch.Tensor):
        return self.model_rep.repr(x_batch)


    def configure_optimizers(self):
        if (self.optimizer_name.lower() == "adam") & (self.setup_optimizer != "cub"):
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=4e-5,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, verbose=True)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # or "step" if you want it to step every batch
                    "frequency": 1,  # Frequency to apply the scheduler
                },
            }
        elif self.setup_optimizer == "cub":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=.90,
                weight_decay=4e-5,
            )
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                verbose=True,
                patience=10,
                factor=0.1,
                min_lr=getattr(self, 'min_lr', 1e-5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler":
                    {
                "scheduler":    lr_scheduler,
                "monitor": "train_loss",
            }}
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                # momentum=.90,
                weight_decay=4e-5
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, verbose=True)
            # import pdb; pdb.set_trace()
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # or "step" if you want it to step every batch
                    "frequency": 1,  # Frequency to apply the scheduler
                },
            }

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is None:
                raise ValueError("Metric must be provided for ReduceLROnPlateau scheduler.")
            scheduler.step(metric)
        else:
            scheduler.step()

    def concept_forward(
        self, x_batch: torch.Tensor
    ):  # something smarter can be done here
        c_logit = []
        x_batch = self.repr(x_batch)
        for module in self.model_concept:
            c_logit.append(module.forward(x_batch))
        return torch.stack(c_logit, dim=1)

    def task_forward(self, c_batch: torch.Tensor):
        y_logit = []
        for module in self.model:
            y_logit.append(module.forward(c_batch))
        return torch.stack(y_logit, dim=1)

    def _vector_loss(
        self,
        y_logit: torch.Tensor,
        y_batch: torch.Tensor,
        y_expert: torch.Tensor | None = None,
        concepts: bool = False,
    ) -> torch.Tensor:
        """
        Computes the loss function.

        Parameters
        ----------
        y_logit : torch.Tensor
            The logits of the network (batch_size, n_classes+n_experts).
        y_batch : torch.Tensor
            The target labels (batch_size, 1).
        y_expert : torch.Tensor
            The expert labels (batch_size, n_experts).
        """
        assert len(y_batch.size()) == 3
        assert len(y_expert.size()) == 3
        assert len(y_logit.size()) == 3
        if concepts:
            self.indicator = 2
            lambdas = self.concept_lambdas.to(self.device)
            self.psi_loss = self.psi_loss_concepts
            if self.weights_concepts is not None:
                assert self.weights_concepts.size() == (1, self.n_concepts, self.indicator)
                weights = self.weights_concepts.to(self.device)
                w_batch = weights.expand(y_batch.shape[0], self.n_concepts, self.indicator)
                w_vec = torch.gather(w_batch, -1, y_batch)
            else:
                w_vec = torch.tensor(1).to(self.device)
        else:
            self.indicator = self.n_classes
            lambdas = self.task_lambdas.to(self.device)
            self.psi_loss = self.psi_loss_tasks
            if self.weights_tasks is not None:
                assert self.weights_tasks.size() == (1, self.n_tasks, self.indicator)
                weights = self.weights_tasks.to(self.device)
                w_batch = weights.expand(y_batch.shape[0], self.n_tasks, self.indicator)
                w_vec = torch.gather(w_batch, -1, y_batch)
            else:
                w_vec = torch.tensor(1).to(self.device)
        mask_right = y_expert == y_batch
        mask_wrong = y_expert != y_batch
        # mask_wrong = mask_wrong.squeeze(1)
        if self.psi_loss == "CE":
            # first term of the loss by Liu et al. 2024 (no label smoothing)
            l_1_all =  (-torch.log_softmax(y_logit, dim=-1))
            l_1 = torch.gather(l_1_all, -1, y_batch)
            #weight = self.weights # tensor 1 * n_concepts l_1 *weight
            # second term for the loss function defined by Liu (still label smoothing)
            if self.label_smoothing_free:
                l_2 = (
                    +(self.cost)
                    * (
                        torch.min(l_1_all[:, :, : self.indicator], dim=-1)
                    ).values.unsqueeze(-1)
                    * mask_wrong.to(torch.float32)
                )
            else:
                l_2 = (
                    +(self.cost)
                    * (torch.sum(l_1_all[:, :, :self.indicator], dim=-1)).unsqueeze(-1)
                    * mask_wrong.to(torch.float32)
                )
            # third term for the loss function defined by Liu (no label smoothing)
            l_3 = (
                +(1 - self.cost)
                * mask_right.to(torch.float32)
                * l_1_all[:, :, self.indicator:]
            )
            loss = (
                torch.mean(l_1 * w_vec * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3 * w_vec * lambdas)
            )
            # self.log("loss", loss)
        elif self.psi_loss == "RS":
            exp_1_all = torch.exp(y_logit)
            exp_gt = torch.gather(exp_1_all, -1, y_batch)
            exp_humans = exp_1_all[:, :, self.indicator :]
            loss_vector = -2 * torch.log(
                (exp_gt + exp_humans * mask_right.to(torch.float32))
                / torch.sum(exp_1_all, dim=-1).unsqueeze(-1)
                + self.eps
            ) * w_vec
            loss = (1 - self.alpha) * torch.mean(loss_vector)
            if self.alpha > 0:
                l_1_all = -torch.log_softmax(y_logit[:, :, : self.indicator], dim=-1)
                l_1 = torch.gather(l_1_all, -1, y_batch) * w_vec
                loss += (self.alpha) * torch.mean(l_1 * lambdas)
            if self.label_smoothing_free:
                raise NotImplementedError(
                    "Label smoothing is not implemented for RS loss."
                )  # TODO
        elif self.psi_loss == "OVA":
            l_1_all_neg = -torch.nn.functional.logsigmoid(
                y_logit
            )  # log(1-exp(-g(x))) on all ys
            l_1_all_pos = -torch.nn.functional.logsigmoid(
                -y_logit
            )  # log(1-exp(g(x))) on all ys
            l_1_part1 = torch.gather(
                l_1_all_neg, -1, y_batch
            )  # log(1-exp(-g(x))) on ground truth
            l_1_part2 = torch.gather(
                l_1_all_pos, -1, y_batch
            )  # log(1-exp(g(x))) on ground truth
            l_1 = (
                l_1_part1 + torch.sum(l_1_all_pos, dim=-1).unsqueeze(-1) - l_1_part2
            )  # log(1-exp(-g(x))) on ground truth + sum(log(1-exp(g(x))) on all ys) - log(1-exp(g(x))) on ground truth
            if self.label_smoothing_free:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        torch.min(
                            l_1_all_neg[:, :, : self.indicator]
                            + torch.sum(l_1_all_pos, dim=-1).unsqueeze(-1)
                            - l_1_all_pos[:, :, : self.indicator],
                            dim=-1,
                        ).values.unsqueeze(
                            -1
                        )  # log(1-exp(-g(x))) on all ys + sum(log(1-exp(g(x))) on all ys) - log(1-exp(g(x))) on al
                    )
                )
            else:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        (
                            torch.sum(
                                l_1_all_neg[:, :, : self.indicator], dim=-1
                            )  # log(1-exp(-g(x))) on all ys
                            + torch.sum(l_1_all_pos[:, :, : self.indicator], dim=-1)
                        )
                        * (
                            self.indicator - self.n_experts
                        )  # log(1-exp(g(x))) on all ys * (n_classes-n_experts)
                        + torch.sum(l_1_all_pos[:, :, self.indicator :], dim=-1)
                        * (self.n_experts)  # log(1-exp(-g(x))) on all n_experts
                    ).unsqueeze(-1)
                )
            l_3 = (
                +mask_right.to(torch.float32)
                * (1 - self.cost)
                * (
                    torch.sum(
                        l_1_all_neg[:, :, self.indicator :]
                        - l_1_all_pos[:, :, self.indicator :],
                        dim=-1,
                    ).unsqueeze(-1)
                )
            )
            loss = (
                torch.mean(l_1 * w_vec * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3*w_vec * lambdas)
            )
        elif self.psi_loss == "ASM":
            # ADD A STABILIZER FOR LOGIT (TO AVOID EXPLOSIONS)
            y_logit = y_logit - torch.max(y_logit, dim=-1).values.unsqueeze(-1)
            logsm_onlyML = -torch.nn.functional.log_softmax(
                y_logit[:, :, : self.indicator], dim=-1
            )  # log softmax only on ML classes
            sm_onlyML_gt = torch.gather(
                logsm_onlyML, -1, y_batch
            )  # ground truth log softmax value
            exp_onlyML = torch.exp(
                y_logit[:, :, : self.indicator]
            )  # exponential values only on ML classes
            exp_all = torch.exp(y_logit)  # exponential values on all logits
            sum_exp_onlyML = torch.sum(exp_onlyML, dim=-1).unsqueeze(
                -1
            )  # sum of exponential values only on ML classes
            sum_exp_all = torch.sum(exp_all, dim=-1).unsqueeze(-1)
            max_exp_onlyML = torch.max(exp_onlyML, dim=-1).values.unsqueeze(
                -1
            )  # max of exponential values only on ML classes
            numerator_1 = (
                    sum_exp_onlyML - max_exp_onlyML
            )  # numerator of the second part (ASM) in Cao et al., 2024
            denominator_1 = (
                    sum_exp_all - max_exp_onlyML + self.eps
            )  # denominator of the second part (ASM) in Cao et al., 2024
            exp_expert = exp_all[:, :, self.indicator:]
            l_1 = ( sm_onlyML_gt - torch.log(
                (numerator_1 / denominator_1) + self.eps
            )
                          )# first part of the loss function in Cao et al., 2024
            sum_all_logsm_onlyML = torch.sum(logsm_onlyML, dim=-1).unsqueeze(
                -1
            )  # sum of log softmax values only on ML classes
            sum_all_part2 = self.indicator * torch.log(
                numerator_1 / denominator_1 + self.eps
            )  # sum of log softmax values on all classes
            if self.label_smoothing_free:
                l_2 = (
                        +mask_wrong.to(torch.float32)
                        * (self.cost)
                        * (
                            torch.min(
                                logsm_onlyML
                                - torch.log(
                                    torch.clamp(1 - (exp_expert / denominator_1), min=self.eps)),
                                dim=-1,
                            ).values.unsqueeze(-1)
                            # second part of the loss function in Cao et al., 2024
                        )
                )
            else:
                l_2 = (
                        +mask_wrong.to(torch.float32)
                        * (self.cost)
                        * (sum_all_logsm_onlyML - sum_all_part2)
                )  # second part of the loss function in Cao et al., 2024

            l_3 = (
                    +mask_right.to(torch.float32)
                    * (1 - self.cost)
                    * (
                            -torch.log(torch.clamp(exp_expert / denominator_1, min=self.eps))
                            + torch.log(torch.clamp(numerator_1 / denominator_1, min=self.eps))
                    )
            )
            # third part of the loss function in Cao et al., 2024
            loss = (
                    torch.mean(w_vec*l_1 * lambdas)
                    + torch.mean(l_2 * lambdas)
                    + torch.mean(w_vec*l_3 * lambdas)
            )
        elif self.psi_loss == "CE-nohuman":
            l_1_all = -torch.log_softmax(y_logit[:, :, : self.indicator], dim=-1)
            l_1 = torch.gather(l_1_all, -1, y_batch)
            loss = torch.mean(w_vec*l_1 * lambdas)
        # import pdb; pdb.set_trace()
        # self.log("loss", loss)
        return loss

    def forward(
        self, x_batch: torch.Tensor, y_expert: torch.Tensor | None = None
    ) -> torch.Tensor:
        if (y_expert is None) or (self.psi_loss == "CE-nohuman"):
            return self.concept_forward(x_batch)
        else:
            c_logit = self.concept_forward(x_batch)
            c_logit = c_logit.view(-1, self.n_concepts, 2 + self.n_experts)
            preds = torch.argmax(c_logit, dim=-1).unsqueeze(-1)
            if self.n_experts > 0:
                assert self.n_experts == 1
                assert y_expert is not None
                # Mask for index of expert
                expert_id = self.n_classes
                mask_expert = preds == expert_id
                # Replace the expert prediction with the expert label
                preds = torch.where(mask_expert, y_expert, preds)
            return self.task_forward(preds)

    def predict_from_logits(
        self, y_logit: torch.Tensor, y_expert: torch.Tensor | None = None, concepts=False
    ) -> torch.Tensor:
        if concepts is True:
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

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # type: ignore
        """Computes the training loss."""

        if self.n_experts == 0:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = None
            c_batch = batch[3]
            c_expert = None
        else:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = batch[2]
            c_batch = batch[3]
            c_expert = batch[4]
        if self.n_tasks == 1:
            y_batch = y_batch.unsqueeze(-1)
        c_batch = c_batch.unsqueeze(-1)
        # Compute Logits (for concepts)
        c_logit = self.concept_forward(x_batch)
        c_logit = c_logit.view(-1, self.n_concepts, 2 + self.n_experts)
        # Compute Logits (for the final task)
        if self.training_mode == "independent":
            y_logit = self.task_forward(c_batch.float().squeeze(-1))
        elif self.training_mode == "sequential":
            c_preds = self.predict_from_logits(c_logit, y_expert=c_expert)
            y_logit = self.task_forward(c_preds.float())
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
            loss_con = self._vector_loss(c_logit, c_batch, c_expert, concepts=True)
            loss_task = self._vector_loss(y_logit, y_batch, y_expert, concepts=False)
            loss_val = loss_con + loss_task
        else:
            raise NotImplementedError(
                "Only one expert is supported. If you want the baseline no expert use psi_loss == 'CE-nohuman'"
            )
        self.log("train_loss", loss_val, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss_val}

    def _run_step(self,batch):
            x_te = batch[0]
            y_te = batch[1]
            y_human = batch[2]
            c_te = batch[3]
            c_human = batch[4]
            # Accuracy (Test)
            c_logit = self.concept_forward(x_te)
            c_logit = c_logit.view(-1, self.n_concepts, 2 + self.n_experts)
            if len(c_human.shape) == 2:
                c_human = c_human.unsqueeze(-1)
            if self.n_experts == 0:
                c_pred = self.predict_from_logits(c_logit, None)
            else:
                c_pred = self.predict_from_logits(c_logit, c_human, concepts=True)
            y_logit = self.task_forward(
                torch.Tensor(c_pred).view(-1, self.n_concepts).float()
            )
            if self.n_experts == 0:
                y_pred = self.predict_from_logits(y_logit, None)
            else:
                y_pred = self.predict_from_logits(y_logit, y_human.view(-1, self.n_tasks, self.n_experts))
            y_preds = y_pred.view(-1, self.n_tasks)
            c_preds = c_pred.view(-1, self.n_concepts)
            return y_preds, c_preds, c_logit, y_logit, c_human, y_human, c_te, y_te
            # c_logits.append(c_logit)
            # y_logits.append(y_logit.view(-1, model.n_tasks, model.n_classes + model.n_experts))
            # y_preds_expert.append(y_human.view(-1, model.n_tasks, model.n_experts))
    def validation_step(self, batch, batch_idx):
        if self.n_experts == 0:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = None
            c_batch = batch[3]
            c_expert = None
        else:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = batch[2]
            c_batch = batch[3]
            c_expert = batch[4]
        if self.n_tasks == 1:
            y_batch = y_batch.unsqueeze(-1)
        c_batch = c_batch.unsqueeze(-1)
        # Compute Logits (for concepts)
        c_logit = self.concept_forward(x_batch)
        c_logit = c_logit.view(-1, self.n_concepts, 2 + self.n_experts)
        # Compute Logits (for the final task)
        if self.training_mode == "independent":
            y_logit = self.task_forward(c_batch.float().squeeze(-1))
        elif self.training_mode == "sequential":
            c_preds = self.predict_from_logits(c_logit, y_expert=c_expert, concepts=True)
            y_logit = self.task_forward(c_preds.float())
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
            loss_con = self._vector_loss(c_logit, c_batch, c_expert, concepts=True)
            loss_task = self._vector_loss(y_logit, y_batch, y_expert, concepts=False)
            loss_val = loss_con + loss_task
        else:
            raise NotImplementedError(
                "Only one expert is supported. If you want the baseline no expert use psi_loss == 'CE-nohuman'"
            )
        self.log("loss_val", loss_val)
        y_preds = self.predict_from_logits(y_logit, y_expert.view(-1, self.n_tasks, self.n_experts))
        c_preds = self.predict_from_logits(c_logit, c_expert, concepts=True)
        acc_test_task = self.accuracy_val(y_preds.view(-1,self.n_tasks), y_batch.view(-1,self.n_tasks))
        acc_test_concepts = self.concept_accuracy_val(c_preds.view(-1, self.n_concepts), c_batch.view(-1,self.n_concepts))
        err_rate_task = (y_preds.view(-1,self.n_tasks) != y_batch.view(-1,self.n_tasks)).float().mean()
        err_rate_concepts = (c_preds.view(-1, self.n_concepts) != c_batch.view(-1,self.n_concepts)).float().mean()
        self.err_c_y_val.append(.5 * (err_rate_task + err_rate_concepts))
        self.err_rate_task_val.append(err_rate_task)
        # self.log("error_rate_val", 1 - acc_test_task, on_step=False, on_epoch=True)
        # self.log("err_c_y_val", self.err_c_y_val, on_step=False, on_epoch=True)
        return loss_val

    # def on_train_epoch_end(self) -> None:
    #     """Log accuracy."""
    #     self.log("train_acc_epoch", self.accuracy, prog_bar=True)
    def on_validation_epoch_end(self) -> None:
        # self.log("acc_val", self.accuracy_val, prog_bar=True)
        avg_error_rate = torch.stack([x for x in self.err_c_y_val]).mean()
        avg_err_task = torch.stack([x for x in self.err_rate_task_val]).mean()
        self.log("err_c_y_val", avg_error_rate, prog_bar=True)
        self.log("err_task_val", avg_err_task, prog_bar=True)
        self.err_rate_task_val.clear()
        self.err_c_y_val.clear()
    def predict_step(self, batch, batch_idx):
        y_preds, c_preds, c_logit, y_logit = self._run_step(batch)
        return {
            "y_preds": y_preds,
            "c_preds": c_preds,
            "c_logit": c_logit,
            "y_logit": y_logit,
        }


class RN34Base(pl.LightningModule):
    """
    A consistent defer classifier is a multi-class
    classifier that has an additional label for each
    of the available experts. The loss function is
    then consistent with the deferment of the expert.
    """

    def __init__(
        self,
        n_classes: int,
        n_tasks: int,
        n_experts: int,
        psi_loss: str = "CE",
        label_smoothing: bool = False,
        cost: float = 0.5,
        learning_rate: float = 1e-3,
        task_lambdas: list = None,
        epsilon=1e-5,
        alpha=0,
    ):
        super(RN34Base, self).__init__()

        # Store internal parameters
        self.n_classes = n_classes
        self.n_experts = n_experts
        self.psi_loss = psi_loss
        self.label_smoothing_free = label_smoothing
        self.learning_rate = learning_rate
        self.cost = cost
        self.n_tasks = n_tasks
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.accuracy_val = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.eps = epsilon
        self.alpha = alpha
        # Check if psi loss is valid
        if psi_loss not in ["CE", "OVA", "ASM", "RS", "ASM", "CE-nohuman"]:
            raise ValueError(f"Invalid psi loss function {psi_loss}.")

        # Check if the number of experts is valid
        if self.n_experts not in [0, 1, 2]:
            raise NotImplementedError("Only one or zero experts are supported.")

        # Check that the cost is valid
        assert self.cost <= 1 and self.cost >= 0

        if task_lambdas is None:
            tmp = [1 for _ in range(n_tasks)]
            self.task_lambdas = torch.ones(
                size=(1, n_tasks, n_experts), device=self.device
            )
            self.task_lambdas[0, :, 0] = torch.tensor(tmp, device=self.device)
        else:
            self.task_lambdas = torch.ones(
                size=(1, n_tasks, n_experts), device=self.device
            )
            self.task_lambdas[0, :, 0] = torch.tensor(task_lambdas, device=self.device)
        # Initialize the model
        self.n_outputs_tasks = (n_classes + n_experts) * n_tasks
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(512, self.n_outputs_tasks)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _psi(self, y_logit: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """
        Computes the learning to defer function, a.k.a., the psi function.
        The tensor y_pred contains the logits for the predicted labels and
        has shape (batch_size, n_classes + n_experts). The tensor y_batch
        has shape (batch_size, 1) and contains the target labels. The function
        returns a tensor of shape (batch_size, 1).
        """
        # check datatype
        assert y_target.dtype == torch.int64

        if self.psi_loss == "CE":
            # Compute the cross-entropy loss
            log_probs = torch.log_softmax(y_logit, dim=1)
            true_log_probs = torch.gather(log_probs, 1, y_target).squeeze(1)
            return -true_log_probs
        else:
            raise NotImplementedError("Only CE is supported.")

    def _vector_loss(
        self,
        y_logit: torch.Tensor,
        y_batch: torch.Tensor,
        y_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the loss function.

        Parameters
        ----------
        y_logit : torch.Tensor
            The logits of the network (batch_size, n_classes+n_experts).
        y_batch : torch.Tensor
            The target labels (batch_size, 1).
        y_expert : torch.Tensor
            The expert labels (batch_size, n_experts).
        """
        assert len(y_batch.size()) == 3
        assert len(y_expert.size()) == 3
        assert len(y_logit.size()) == 3
        self.indicator = self.n_classes
        lambdas = self.task_lambdas.to(self.device)  # 1 x n_tasks x 1
        mask_right = y_expert == y_batch
        # mask_right = mask_right.squeeze(1)
        # double check when increasing the
        mask_wrong = y_expert != y_batch
        # mask_wrong = mask_wrong.squeeze(1)
        if self.psi_loss == "CE":
            # first term of the loss by Liu et al. 2024 (no label smoothing)
            l_1_all = -torch.log_softmax(y_logit, dim=-1)
            l_1 = torch.gather(l_1_all, -1, y_batch)
            # second term for the loss function defined by Liu (still label smoothing)
            if self.label_smoothing_free:
                l_2 = (
                    +(self.cost)
                    * (
                        torch.min(l_1_all[:, :, : self.indicator], dim=-1)
                    ).values.unsqueeze(-1)
                    * mask_wrong.to(torch.float32)
                )
            else:
                l_2 = (
                    +(self.cost)
                    * (torch.sum(l_1_all[:, :, : self.indicator], dim=-1)).unsqueeze(-1)
                    * mask_wrong.to(torch.float32)
                )
            # third term for the loss function defined by Liu (no label smoothing)
            l_3 = (
                +(1 - self.cost)
                * mask_right.to(torch.float32)
                * l_1_all[:, :, self.indicator :]
            )
            loss = (
                torch.mean(l_1 * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3 * lambdas)
            )
            # self.log("loss", loss)
        elif self.psi_loss == "RS":
            exp_1_all = torch.exp(y_logit)
            exp_gt = torch.gather(exp_1_all, -1, y_batch)
            exp_humans = exp_1_all[:, :, self.indicator :]
            loss_vector = -2 * torch.log(
                (exp_gt + exp_humans * mask_right.to(torch.float32))
                / torch.sum(exp_1_all, dim=-1).unsqueeze(-1)
                + self.eps
            )
            loss = (1 - self.alpha) * torch.mean(loss_vector)
            if self.alpha > 0:
                l_1_all = -torch.log_softmax(y_logit[:, :, : self.indicator], dim=-1)
                l_1 = torch.gather(l_1_all, -1, y_batch)
                loss += (self.alpha) * torch.mean(l_1 * lambdas)
            if self.label_smoothing_free:
                raise NotImplementedError(
                    "Label smoothing is not implemented for RS loss."
                )  # TODO
        elif self.psi_loss == "OVA":
            l_1_all_neg = -torch.nn.functional.logsigmoid(
                y_logit
            )  # log(1-exp(-g(x))) on all ys
            l_1_all_pos = -torch.nn.functional.logsigmoid(
                -y_logit
            )  # log(1-exp(g(x))) on all ys
            l_1_part1 = torch.gather(
                l_1_all_neg, -1, y_batch
            )  # log(1-exp(-g(x))) on ground truth
            l_1_part2 = torch.gather(
                l_1_all_pos, -1, y_batch
            )  # log(1-exp(g(x))) on ground truth
            l_1 = (
                l_1_part1 + torch.sum(l_1_all_pos, dim=-1).unsqueeze(-1) - l_1_part2
            )  # log(1-exp(-g(x))) on ground truth + sum(log(1-exp(g(x))) on all ys) - log(1-exp(g(x))) on ground truth
            if self.label_smoothing_free:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        torch.min(
                            l_1_all_neg[:, :, : self.indicator]
                            + torch.sum(l_1_all_pos, dim=-1).unsqueeze(-1)
                            - l_1_all_pos[:, :, : self.indicator],
                            dim=-1,
                        ).values.unsqueeze(
                            -1
                        )  # log(1-exp(-g(x))) on all ys + sum(log(1-exp(g(x))) on all ys) - log(1-exp(g(x))) on al
                    )
                )
            else:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        (
                            torch.sum(
                                l_1_all_neg[:, :, : self.indicator], dim=-1
                            )  # log(1-exp(-g(x))) on all ys
                            + torch.sum(l_1_all_pos[:, :, : self.indicator], dim=-1)
                        )
                        * (
                            self.indicator - self.n_experts
                        )  # log(1-exp(g(x))) on all ys * (n_classes-n_experts)
                        + torch.sum(l_1_all_pos[:, :, self.indicator :], dim=-1)
                        * (self.n_experts)  # log(1-exp(-g(x))) on all n_experts
                    ).unsqueeze(-1)
                )
            l_3 = (
                +mask_right.to(torch.float32)
                * (1 - self.cost)
                * (
                    torch.sum(
                        l_1_all_neg[:, :, self.indicator :]
                        - l_1_all_pos[:, :, self.indicator :],
                        dim=-1,
                    ).unsqueeze(-1)
                )
            )
            loss = (
                torch.mean(l_1 * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3 * lambdas)
            )
        elif self.psi_loss == "ASM":
            logsm_onlyML = -torch.nn.functional.log_softmax(
                y_logit[:, :, : self.indicator], dim=-1
            )  # log softmax only on ML classes
            sm_onlyML_gt = torch.gather(
                logsm_onlyML, -1, y_batch
            )  # ground truth log softmax value
            exp_onlyML = torch.exp(
                y_logit[:, :, : self.indicator]
            )  # exponential values only on ML classes
            exp_all = torch.exp(y_logit)  # exponential values on all logits
            sum_exp_onlyML = torch.sum(exp_onlyML, dim=-1).unsqueeze(
                -1
            )  # sum of exponential values only on ML classes
            sum_exp_all = torch.sum(exp_all, dim=-1).unsqueeze(-1)
            max_exp_onlyML = torch.max(exp_onlyML, dim=-1).values.unsqueeze(
                -1
            )  # max of exponential values only on ML classes
            numerator_1 = (
                sum_exp_onlyML - max_exp_onlyML
            )  # numerator of the second part (ASM) in Cao et al., 2024
            denominator_1 = (
                sum_exp_all - max_exp_onlyML + self.eps
            )  # denominator of the second part (ASM) in Cao et al., 2024
            exp_expert = exp_all[:, :, self.indicator :]
            l_1 = sm_onlyML_gt - torch.log(
                (numerator_1 / denominator_1) + self.eps
            )  # first part of the loss function in Cao et al., 2024
            sum_all_logsm_onlyML = torch.sum(logsm_onlyML, dim=-1).unsqueeze(
                -1
            )  # sum of log softmax values only on ML classes
            sum_all_part2 = self.indicator * torch.log(
                numerator_1 / denominator_1 + self.eps
            )  # sum of log softmax values on all classes
            if self.label_smoothing_free:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        torch.min(
                            logsm_onlyML
                            - torch.log(1 - (exp_expert / denominator_1) + self.eps),
                            dim=-1,
                        ).values.unsqueeze(-1)
                        # second part of the loss function in Cao et al., 2024
                    )
                )
            else:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (sum_all_logsm_onlyML - sum_all_part2)
                )  # second part of the loss function in Cao et al., 2024

            l_3 = (
                +mask_right.to(torch.float32)
                * (1 - self.cost)
                * (
                    -torch.log(exp_expert / denominator_1 + self.eps)
                    + torch.log((numerator_1 / denominator_1) + self.eps)
                )
            )
            # third part of the loss function in Cao et al., 2024
            loss = (
                torch.mean(l_1 * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3 * lambdas)
            )
        elif self.psi_loss == "CE-nohuman":
            l_1_all = -torch.log_softmax(y_logit[:, :, : self.indicator], dim=-1)
            l_1 = torch.gather(l_1_all, -1, y_batch)
            loss = torch.mean(l_1 * lambdas)
        self.log("loss", loss)
        return loss

    def forward(self, x_batch: torch.Tensor):
        return self.model.forward(x_batch)

    def predict_from_logits(
        self, y_logit: torch.Tensor, y_expert: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Compute the predictions
        preds = torch.argmax(y_logit[:,:self.n_classes], dim=-1).unsqueeze(1)
        return preds

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # type: ignore
        """Computes the training loss."""

        if self.n_experts == 0:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = None
        else:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = batch[2]

        # Compute Logits
        y_logit = self.model.forward(x_batch)
        y_logit = y_logit.view(-1, self.n_tasks, self.n_classes + self.n_experts)
        # Compute the loss
        # loss_val = self._loss(y_logit, y_batch, y_expert=y_expert)
        if self.n_tasks == 1:
            y_batch = y_batch.unsqueeze(-1)
        if self.n_experts > 0:
            loss_val = 0
            y_expert = y_expert.view(-1, self.n_tasks, self.n_experts)
            loss_val = self._vector_loss(y_logit, y_batch, y_expert)
        self.log("loss", loss_val)
        if self.n_experts == 0:
            preds = self.predict_from_logits(y_logit, y_expert=None)
        else:
            preds = self.predict_from_logits(y_logit, y_expert=y_expert)
        acc_val = self.accuracy(preds, y_batch)
        self.log("train_acc", acc_val)

        return loss_val

    def predict_step(self,batch):
            y_logit = self.forward(batch[0]
            )
            if self.n_experts == 0:
                y_pred = self.predict_from_logits(y_logit, None)
            else:
                y_pred = self.predict_from_logits(y_logit, batch[2].view(-1, self.n_tasks, self.n_experts))
            y_preds = y_pred.view(-1, self.n_tasks)
            return y_preds, y_logit
            # c_logits.append(c_logit)
            # y_logits.append(y_logit.view(-1, model.n_tasks, model.n_classes + model.n_experts))
            # y_preds_expert.append(y_human.view(-1, model.n_tasks, model.n_experts))
    def validation_step(self, batch):
        y_true = batch[1]
        c_true = batch[3]
        y_preds, y_logit = self.predict_step(batch)
        acc_test_task = self.accuracy_val(y_preds, y_true)
        result = {"acc_test_task": acc_test_task,  "error_rate": 1-acc_test_task}
        return result

    def on_train_epoch_end(self) -> None:
        """Log accuracy."""
        self.log("train_acc_epoch", self.accuracy, prog_bar=True)
    def on_validation_epoch_end(self) -> None:
        self.log("val_acc_epoch", self.accuracy_val, prog_bar=True)

class BBConsistentVision(pl.LightningModule):
    """
    The BlackBox consistent model
    """

    def __init__(
        self,
        model_rep: torch.nn.Module,
        emb_size: int,
        n_classes: int,
        n_tasks: int,
        n_experts: int,
        psi_loss: str = "CE",
        label_smoothing: bool = False,
        cost: float = 0.5,
        learning_rate: float = 1e-2,
        training_mode: str = "independent",
        concept_lambdas: list = None,
        task_lambdas: list = None,
        epsilon: float = 1e-8,
        alpha: float = 0,
        psi_loss_tasks=None,
        weights_concepts=None,
        weights_tasks=None,
        optimizer_name="adamW",
        halved: bool = False,
        setup_optimizer: str = "cifar10",
        pretrain_freeze: bool = True
    ):
        super(BBConsistentVision, self).__init__()
        self.n_classes = n_classes  # for now the same for all final_tasks
        self.n_tasks = n_tasks  # different tasks possible
        self.n_experts = (
            n_experts  # for now same number of experts for each concept/task
        )
        self.psi_loss = psi_loss
        if psi_loss_tasks is None:
            self.psi_loss_tasks = psi_loss
        else:
            self.psi_loss_tasks = psi_loss_tasks
        self.label_smoothing_free = label_smoothing
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.cost = cost  # to decide how to implement it for multi concpets/tasks and multi experts
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.accuracy_val = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.eps = epsilon
        if self.psi_loss == "ASM":
            if self.eps < 1e-5:
                self.eps = 1e-5
        self.alpha = alpha
        if task_lambdas is None:
            tmp = [1 for _ in range(n_tasks)]
            self.task_lambdas = torch.ones(size=(1, n_tasks, 1), device=self.device)
            self.task_lambdas[0, :, 0] = torch.tensor(tmp, device=self.device)
        else:
            self.task_lambdas = torch.ones(size=(1, n_tasks, 1), device=self.device)
            self.task_lambdas[0, :, 0] = torch.tensor(task_lambdas, device=self.device)
        # Check if psi loss is valid
        if self.psi_loss not in ["CE", "OVA", "ASM", "ASM", "RS", "CE-nohuman"]:
            raise ValueError(f"Invalid psi loss function {psi_loss} not in CE.")
        if self.psi_loss_tasks not in ["CE", "OVA", "ASM", "ASM", "RS", "CE-nohuman"]:
            raise ValueError(
                f"Invalid psi loss function {self.psi_loss_tasks} not in CE."
            )
        # Check if the number of experts is valid
        if self.n_experts not in [0, 1]:
            raise NotImplementedError("Only one or zero experts are supported.")
        # Check that the cost is valid
        assert self.cost <= 1 and self.cost >= 0
        self.epsilon = 1e-8
        self.weights_tasks = weights_tasks
        # Initialize the model
        self.n_outputs_tasks = (n_classes + n_experts) * n_tasks
        self.model_rep = model_rep
        assert hasattr(self.model_rep, "repr")
        if type(model_rep) == WideResNet:
            self.in_features = self.model_rep.fc2.in_features
        elif type(model_rep) == InceptionV3:
            self.in_features = self.model_rep.fc.in_features
        elif type(model_rep) == Resn34:
            self.in_features = self.model_rep.fc.in_features
        self.model = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.in_features, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, n_classes + n_experts),
                )
                for _ in range(n_tasks)
            ]
        )
        self.training_mode = training_mode
        self.err_rate_task_val = []
        self.setup_optimizer = setup_optimizer

    def repr(self, x_batch: torch.Tensor):
        return self.model_rep.repr(x_batch)
    def configure_optimizers(self):
        if (self.optimizer_name.lower() == "adam") & (self.setup_optimizer != "cub"):
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=4e-5,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, verbose=True)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # or "step" if you want it to step every batch
                    "frequency": 1,  # Frequency to apply the scheduler
                },
            }
        elif self.setup_optimizer == "cub":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=.90,
                weight_decay=4e-5,
            )
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                verbose=True,
                patience=10,
                factor=0.1,
                min_lr=getattr(self, 'min_lr', 1e-5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler":
                    {
                "scheduler":    lr_scheduler,
                "monitor": "train_loss",
            }}
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                # momentum=.90,
                weight_decay=4e-5
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, verbose=True)
            # import pdb; pdb.set_trace()
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # or "step" if you want it to step every batch
                    "frequency": 1,  # Frequency to apply the scheduler
                },
            }

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is None:
                raise ValueError("Metric must be provided for ReduceLROnPlateau scheduler.")
            scheduler.step(metric)
        else:
            scheduler.step()

    def task_forward(self, x_batch: torch.Tensor):  # something smarter can be done here
        y_logit = []
        x_batch = self.repr(x_batch)
        for module in self.model:
            y_logit.append(module.forward(x_batch))
        return torch.stack(y_logit, dim=1)

    def _vector_loss(
        self,
        y_logit: torch.Tensor,
        y_batch: torch.Tensor,
        y_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the loss function.

        Parameters
        ----------
        y_logit : torch.Tensor
            The logits of the network (batch_size, n_classes+n_experts).
        y_batch : torch.Tensor
            The target labels (batch_size, 1).
        y_expert : torch.Tensor
            The expert labels (batch_size, n_experts).
        """
        assert len(y_batch.size()) == 3
        assert len(y_expert.size()) == 3
        assert len(y_logit.size()) == 3
        self.indicator = self.n_classes
        lambdas = self.task_lambdas.to(self.device)
        self.psi_loss = self.psi_loss_tasks
        if self.weights_tasks is not None:
            assert self.weights_tasks.size() == (1, self.n_tasks, self.indicator)
            weights = self.weights_tasks.to(self.device)
            w_batch = weights.expand(y_batch.shape[0])
            w_vec = torch.gather(w_batch, -1, y_batch)
        else:
            w_vec = torch.tensor(1).to(self.device)
        mask_right = y_expert == y_batch
        mask_wrong = y_expert != y_batch
        # mask_wrong = mask_wrong.squeeze(1)
        if self.psi_loss == "CE":
            # first term of the loss by Liu et al. 2024 (no label smoothing)
            l_1_all =  (-torch.log_softmax(y_logit, dim=-1))
            l_1 = torch.gather(l_1_all, -1, y_batch)
            #weight = self.weights # tensor 1 * n_concepts l_1 *weight
            # second term for the loss function defined by Liu (still label smoothing)
            if self.label_smoothing_free:
                l_2 = (
                    +(self.cost)
                    * (
                        torch.min(l_1_all[:, :, : self.indicator], dim=-1)
                    ).values.unsqueeze(-1)
                    * mask_wrong.to(torch.float32)
                )
            else:
                l_2 = (
                    +(self.cost)
                    * (torch.sum(l_1_all[:, :, :self.indicator], dim=-1)).unsqueeze(-1)
                    * mask_wrong.to(torch.float32)
                )
            # third term for the loss function defined by Liu (no label smoothing)
            l_3 = (
                +(1 - self.cost)
                * mask_right.to(torch.float32)
                * l_1_all[:, :, self.indicator:]
            )
            loss = (
                torch.mean(l_1 * w_vec * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3 * w_vec * lambdas)
            )
            # self.log("loss", loss)
        elif self.psi_loss == "RS":
            exp_1_all = torch.exp(y_logit)
            exp_gt = torch.gather(exp_1_all, -1, y_batch)
            exp_humans = exp_1_all[:, :, self.indicator :]
            loss_vector = -2 * torch.log(
                (exp_gt + exp_humans * mask_right.to(torch.float32))
                / torch.sum(exp_1_all, dim=-1).unsqueeze(-1)
                + self.eps
            )
            loss = (1 - self.alpha) * torch.mean(loss_vector)
            if self.alpha > 0:
                l_1_all = -torch.log_softmax(y_logit[:, :, : self.indicator], dim=-1)
                l_1 = torch.gather(l_1_all, -1, y_batch)* w_vec
                loss += (self.alpha) * torch.mean(l_1 * lambdas)
            if self.label_smoothing_free:
                raise NotImplementedError(
                    "Label smoothing is not implemented for RS loss."
                )  # TODO
        elif self.psi_loss == "OVA":
            l_1_all_neg = -torch.nn.functional.logsigmoid(
                y_logit
            )  # log(1-exp(-g(x))) on all ys
            l_1_all_pos = -torch.nn.functional.logsigmoid(
                -y_logit
            )  # log(1-exp(g(x))) on all ys
            l_1_part1 = torch.gather(
                l_1_all_neg, -1, y_batch
            )  # log(1-exp(-g(x))) on ground truth
            l_1_part2 = torch.gather(
                l_1_all_pos, -1, y_batch
            )  # log(1-exp(g(x))) on ground truth
            l_1 = (
                l_1_part1 + torch.sum(l_1_all_pos, dim=-1).unsqueeze(-1) - l_1_part2
            )  # log(1-exp(-g(x))) on ground truth + sum(log(1-exp(g(x))) on all ys) - log(1-exp(g(x))) on ground truth
            if self.label_smoothing_free:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        torch.min(
                            l_1_all_neg[:, :, : self.indicator]
                            + torch.sum(l_1_all_pos, dim=-1).unsqueeze(-1)
                            - l_1_all_pos[:, :, : self.indicator],
                            dim=-1,
                        ).values.unsqueeze(
                            -1
                        )  # log(1-exp(-g(x))) on all ys + sum(log(1-exp(g(x))) on all ys) - log(1-exp(g(x))) on al
                    )
                )
            else:
                l_2 = (
                    +mask_wrong.to(torch.float32)
                    * (self.cost)
                    * (
                        (
                            torch.sum(
                                l_1_all_neg[:, :, : self.indicator], dim=-1
                            )  # log(1-exp(-g(x))) on all ys
                            + torch.sum(l_1_all_pos[:, :, : self.indicator], dim=-1)
                        )
                        * (
                            self.indicator - self.n_experts
                        )  # log(1-exp(g(x))) on all ys * (n_classes-n_experts)
                        + torch.sum(l_1_all_pos[:, :, self.indicator :], dim=-1)
                        * (self.n_experts)  # log(1-exp(-g(x))) on all n_experts
                    ).unsqueeze(-1)
                )
            l_3 = (
                +mask_right.to(torch.float32)
                * (1 - self.cost)
                * (
                    torch.sum(
                        l_1_all_neg[:, :, self.indicator :]
                        - l_1_all_pos[:, :, self.indicator :],
                        dim=-1,
                    ).unsqueeze(-1)
                )
            )
            loss = (
                torch.mean(l_1 * w_vec * lambdas)
                + torch.mean(l_2 * lambdas)
                + torch.mean(l_3*w_vec * lambdas)
            )
        elif self.psi_loss == "ASM":
            # ADD A STABILIZER FOR LOGIT (TO AVOID EXPLOSIONS)
            y_logit = y_logit - torch.max(y_logit, dim=-1).values.unsqueeze(-1)
            logsm_onlyML = -torch.nn.functional.log_softmax(
                y_logit[:, :, : self.indicator], dim=-1
            )  # log softmax only on ML classes
            sm_onlyML_gt = torch.gather(
                logsm_onlyML, -1, y_batch
            )  # ground truth log softmax value
            exp_onlyML = torch.exp(
                y_logit[:, :, : self.indicator]
            )  # exponential values only on ML classes
            exp_all = torch.exp(y_logit)  # exponential values on all logits
            sum_exp_onlyML = torch.sum(exp_onlyML, dim=-1).unsqueeze(
                -1
            )  # sum of exponential values only on ML classes
            sum_exp_all = torch.sum(exp_all, dim=-1).unsqueeze(-1)
            max_exp_onlyML = torch.max(exp_onlyML, dim=-1).values.unsqueeze(
                -1
            )  # max of exponential values only on ML classes
            numerator_1 = (
                    sum_exp_onlyML - max_exp_onlyML
            )  # numerator of the second part (ASM) in Cao et al., 2024
            denominator_1 = (
                    sum_exp_all - max_exp_onlyML + self.eps
            )  # denominator of the second part (ASM) in Cao et al., 2024
            exp_expert = exp_all[:, :, self.indicator:]
            l_1 = ( sm_onlyML_gt - torch.log(
                (numerator_1 / denominator_1) + self.eps
            )
                          )# first part of the loss function in Cao et al., 2024
            sum_all_logsm_onlyML = torch.sum(logsm_onlyML, dim=-1).unsqueeze(
                -1
            )  # sum of log softmax values only on ML classes
            sum_all_part2 = self.indicator * torch.log(
                numerator_1 / denominator_1 + self.eps
            )  # sum of log softmax values on all classes
            if self.label_smoothing_free:
                l_2 = (
                        +mask_wrong.to(torch.float32)
                        * (self.cost)
                        * (
                            torch.min(
                                logsm_onlyML
                                - torch.log(
                                    torch.clamp(1 - (exp_expert / denominator_1), min=self.eps)),
                                dim=-1,
                            ).values.unsqueeze(-1)
                            # second part of the loss function in Cao et al., 2024
                        )
                )
            else:
                l_2 = (
                        +mask_wrong.to(torch.float32)
                        * (self.cost)
                        * (sum_all_logsm_onlyML - sum_all_part2)
                )  # second part of the loss function in Cao et al., 2024

            l_3 = (
                    +mask_right.to(torch.float32)
                    * (1 - self.cost)
                    * (
                            -torch.log(torch.clamp(exp_expert / denominator_1, min=self.eps))
                            + torch.log(torch.clamp(numerator_1 / denominator_1, min=self.eps))
                    )
            )
            # third part of the loss function in Cao et al., 2024
            loss = (
                    torch.mean(w_vec*l_1 * lambdas)
                    + torch.mean(l_2 * lambdas)
                    + torch.mean(w_vec*l_3 * lambdas)
            )
        elif self.psi_loss == "CE-nohuman":
            l_1_all = -torch.log_softmax(y_logit[:, :, : self.indicator], dim=-1)
            l_1 = torch.gather(l_1_all, -1, y_batch)
            loss = torch.mean(w_vec*l_1 * lambdas)
        # import pdb; pdb.set_trace()
        # self.log("loss", loss)
        return loss

    def forward(
        self, x_batch: torch.Tensor, y_expert: torch.Tensor | None = None
    ) -> torch.Tensor:
            return self.task_forward(x_batch)

    def predict_from_logits(
        self, y_logit: torch.Tensor, y_expert: torch.Tensor | None = None, concepts=False
    ) -> torch.Tensor:
        index_expert = self.n_classes
        if (self.psi_loss_tasks == "CE-nohuman"):
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

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # type: ignore
        """Computes the training loss."""

        if self.n_experts == 0:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = None
        else:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = batch[2]
        if self.n_tasks == 1:
            y_batch = y_batch.unsqueeze(-1)
        y_logit = self.task_forward(x_batch)
        if self.n_experts > 0:
            y_expert = y_expert.view(-1, self.n_tasks, self.n_experts)
            loss_task = self._vector_loss(y_logit, y_batch, y_expert)
            loss_val = loss_task
        else:
            raise NotImplementedError(
                "Only one expert is supported. If you want the baseline no expert use psi_loss == 'CE-nohuman'"
            )
        self.log("train_loss", loss_val, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss_val}

    def _run_step(self,batch):
            x_te = batch[0]
            y_te = batch[1]
            y_human = batch[2]
            c_te = batch[3]
            c_human = batch[4]
            # Accuracy (Test)
            y_logit = self.task_forward(x_te
            )
            if self.n_experts == 0:
                y_pred = self.predict_from_logits(y_logit, None)
            else:
                y_pred = self.predict_from_logits(y_logit, y_human.view(-1, self.n_tasks, self.n_experts))
            y_preds = y_pred.view(-1, self.n_tasks)
            return y_preds, y_logit
            # c_logits.append(c_logit)
            # y_logits.append(y_logit.view(-1, model.n_tasks, model.n_classes + model.n_experts))
            # y_preds_expert.append(y_human.view(-1, model.n_tasks, model.n_experts))
    def validation_step(self, batch, batch_idx):
        if self.n_experts == 0:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = None
            c_batch = batch[3]
            c_expert = None
        else:
            x_batch = batch[0]
            y_batch = batch[1]
            y_expert = batch[2]
            c_batch = batch[3]
            c_expert = batch[4]
        if self.n_tasks == 1:
            y_batch = y_batch.unsqueeze(-1)
        y_logit = self.task_forward(x_batch)
        if self.n_experts > 0:
            y_expert = y_expert.view(-1, self.n_tasks, self.n_experts)
            loss_task = self._vector_loss(y_logit, y_batch, y_expert)
            loss_val = loss_task
        else:
            raise NotImplementedError(
                "Only one expert is supported. If you want the baseline no expert use psi_loss == 'CE-nohuman'"
            )
        self.log("loss_val", loss_val)
        y_preds = self.predict_from_logits(y_logit, y_expert.view(-1, self.n_tasks, self.n_experts))
        acc_test_task = self.accuracy_val(y_preds.view(-1,self.n_tasks), y_batch.view(-1,self.n_tasks))
        err_rate_task = (y_preds.view(-1,self.n_tasks) != y_batch.view(-1,self.n_tasks)).float().mean()
        self.err_rate_task_val.append(err_rate_task)
        return loss_val

    # def on_train_epoch_end(self) -> None:
    #     """Log accuracy."""
    #     self.log("train_acc_epoch", self.accuracy, prog_bar=True)
    def on_validation_epoch_end(self) -> None:
        # self.log("acc_val", self.accuracy_val, prog_bar=True)
        self.err_rate_task_val.clear()


class BlackBox(pl.LightningModule):
    def __init__(self,
        model,
        n_classes,
        n_concepts,
        learning_rate: float = 1e-3,
        task: str = "x2y",
        weights_tasks=None,
        n_tasks=1
    ):
        super(BlackBox, self).__init__()
        self.model_rep = model
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_concepts = n_concepts
        self.accuracy_val = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.err_rate_task_val = []
        self.task = task
        self.weights_tasks = weights_tasks
        self.n_tasks = n_tasks
    def _vector_loss(self, y_logit, y_batch):
        if self.weights_tasks is not None:
            assert self.weights_tasks.size() == (1, self.n_tasks, self.n_classes)
            weights = self.weights_tasks.to(self.device)
            w_batch = weights.expand(y_batch.shape[0], self.n_tasks, self.n_classes)
            w_vec = torch.gather(w_batch, -1, y_batch.view(-1, self.n_tasks, 1))
        else:
            w_vec = torch.tensor(1).to(self.device)
        l_1_all = -torch.log_softmax(y_logit, dim=-1)
        if self.task == "x2c":
            l_1 = torch.gather(l_1_all, -1, y_batch.view(-1, self.n_concepts, 1))
        elif self.task == "x2y":
            l_1_all = l_1_all.view(-1, self.n_tasks, self.n_classes)
            l_1 = torch.gather(l_1_all, -1, y_batch.view(-1, self.n_tasks, 1))
        loss = torch.mean(l_1 * w_vec)
        return loss
    def _compute_loss(self, y_logit, y_batch):
        if self.weights_tasks is not None:
            assert self.weights_tasks.size() == (1, self.n_tasks, self.n_classes)
            if self.n_tasks == 1:
                weights = self.weights_tasks.to(y_logit.device)
                weights_shrinked = weights.view(-1)
                loss = torch.nn.functional.cross_entropy(y_logit.view(-1, self.n_classes), y_batch.view(-1),
                                                         weight=weights_shrinked)
                return loss
        else:
            if self.n_tasks == 1:
                loss = torch.nn.functional.cross_entropy(y_logit.view(-1, self.n_classes), y_batch.view(-1),
                                                         )
                return loss


    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # type: ignore
        """Computes the training loss."""

        x_batch = batch[0]
        y_batch = batch[1]
        # Compute Logits
        y_logit = self.model_rep(x_batch)
        if self.task == "x2y":
            y_logit = y_logit.view(-1, self.n_classes)
        elif self.task == "x2c":
            y_logit = y_logit.view(-1, self.n_concepts,  self.n_classes)
        # Compute the loss
        # loss_val = self._loss(y_logit, y_batch, y_expert=y_expert)
        # import pdb; pdb.set_trace()
        loss_val = self._compute_loss(y_logit, y_batch)
        self.log("loss", loss_val, prog_bar=True)

        return loss_val
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model_rep(*args, **kwargs)
    def predict_from_logits(
        self, y_logit: torch.Tensor
    ) -> torch.Tensor:
        # Compute the predictions
        preds = torch.argmax(y_logit, dim=-1).unsqueeze(-1)
        return preds
    def predict_proba_from_logits(self, y_logit: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(y_logit, dim=-1)
    def predict_step(self,batch):
            y_logit = self.forward(batch[0]
            )
            if self.task == "x2y":
                y_logit = y_logit.view(-1, self.n_classes)
            elif self.task == "x2c":
                y_logit = y_logit.view(-1, self.n_concepts,  self.n_classes)
            y_preds = self.predict_from_logits(y_logit)
            return y_preds, y_logit
    def validation_step(self, batch):
        y_true = batch[1]
        y_preds, y_logit = self.predict_step(batch)
        # import pdb; pdb.set_trace()
        if self.task == "x2y":
            acc_test_task = self.accuracy_val(y_preds, y_true.view(-1,1))
        elif self.task == "x2c":
            acc_test_task = self.accuracy_val(y_preds, y_true.view(-1,self.n_concepts, 1))
        result = {"acc_test_task": acc_test_task,  "error_rate": 1-acc_test_task}
        self.log("loss_val", self._compute_loss(y_logit, y_true), prog_bar=True)
        return result
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model_rep.parameters(),
            lr=self.learning_rate,
            # momentum=.90,
            weight_decay=4e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, verbose=True)
        # import pdb; pdb.set_trace()
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or "step" if you want it to step every batch
                "frequency": 1,  # Frequency to apply the scheduler
            },
        }
    #
    # def on_validation_epoch_end(self) -> None:
    #     # self.log("acc_val", self.accuracy_val, prog_bar=True)
    #     # avg_err_task = torch.stack([x for x in self.err_rate_task_val]).mean()
    #     # self.log("err_task_val", avg_err_task, prog_bar=True)
    #     # self.err_rate_task_val.clear()