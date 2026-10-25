import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from DCBM import DCBM
import torch
import pandas as pd


def test_model(
        model,
        testloader,
        seed=42,
        path_results="results.csv",
        also_scores=False,
):
    model.eval()
    if isinstance(model, DCBM):
        y_true = []
        c_true = []
        y_preds = []
        c_preds = []
        c_logits = []
        y_logits = []
        y_preds_expert = []
        # c_test_expert = []
        with torch.no_grad():
            for j, batch in enumerate(testloader):
                x_te = batch[0].to(model.device)
                y_te = batch[1].to(model.device)
                y_human = batch[2].to(model.device)
                c_te = batch[3].to(model.device)
                c_human = batch[4].to(model.device)
                # Accuracy (Test)
                c_logit = model.concept_forward(x_te)
                c_logit = c_logit.view(-1, model.n_concepts, 2 + model.n_experts)
                coverage_concept_dict = {}
                accuracy_concept_dict = {}
                if len(c_human.shape) == 2:
                    c_human = c_human.unsqueeze(-1)
                if model.n_experts == 0:
                    c_pred = model.predict_from_logits(c_logit, None)
                else:
                    c_pred = model.predict_from_logits(c_logit, c_human, concepts=True)
                y_logit = model.task_forward(
                    torch.Tensor(c_pred).view(-1, model.n_concepts).float()
                )
                if model.n_experts == 0:
                    y_pred = model.predict_from_logits(y_logit, None)
                else:
                    y_pred = model.predict_from_logits(y_logit, y_human.view(-1, model.n_tasks, model.n_experts))

                y_true.append(y_te.view(-1, model.n_tasks))
                c_true.append(c_te.view(-1, model.n_concepts))
                y_preds.append(y_pred.view(-1, model.n_tasks))
                c_preds.append(c_pred.view(-1, model.n_concepts))
                c_logits.append(c_logit)
                y_logits.append(y_logit.view(-1, model.n_tasks, model.n_classes + model.n_experts))
                y_preds_expert.append(y_human.view(-1, model.n_tasks, model.n_experts))
        y_preds = torch.cat(y_preds, dim=0).cpu()
        c_preds = torch.cat(c_preds, dim=0).cpu()
        y_true = torch.cat(y_true, dim=0).cpu()
        c_true = torch.cat(c_true, dim=0).cpu()
        c_logits = torch.cat(c_logits, dim=0).cpu()
        y_logits = torch.cat(y_logits, dim=0).cpu()
        y_preds_expert = torch.cat(y_preds_expert, dim=0).cpu()
        coverage_concept_dict = {}
        accuracy_concept_dict = {}
        if model.psi_loss_concepts == "CE-nohuman":
            deferred_concepts = torch.zeros(c_true.size()).unsqueeze(-1)
        else:
            # NOTE: c_logits (all batches), not the c_logit loop leftover -- otherwise
            # coverage is measured on the last batch while accuracy uses every sample.
            deferred_concepts = torch.where(
                torch.argmax(c_logits, dim=2) >= 2, 1, 0
            ).unsqueeze(-1)
        for i in range(model.n_concepts):
            coverage_concept_dict["C_{}".format(i)] = 1 - torch.mean(
                deferred_concepts[:, i].float()
            )
            accuracy_concept_dict["C_{}".format(i)] = accuracy_score(
                c_true[:, i], c_preds[:, i]
            )
        coverage_concepts = 1 - torch.mean(deferred_concepts.float())
        if len(y_logits.size()) > 2:
            if model.psi_loss_tasks == "CE-nohuman":
                deferred_task = torch.zeros(y_true.size()).unsqueeze(-1)
                coverage_task = 1 - torch.mean((deferred_task).float())
            else:
                deferred_task = torch.where(
                    torch.argmax(y_logits, dim=2) == model.n_classes, 1, 0
                )
                coverage_task = 1 - torch.mean(deferred_task.float())
        else:
            if model.psi_loss_tasks == "CE-nohuman":
                deferred_task = torch.zeros(y_true.size()).unsqueeze(-1)
            else:
                deferred_task = torch.where(
                    torch.argmax(y_logits, dim=1) == model.n_classes, 1, 0
                )
            coverage_task = 1 - torch.mean(deferred_task.float())
        acc_test_task = accuracy_score(y_true, y_preds.view(-1, model.n_tasks))
        acc_test_concepts = np.mean(np.array([v for k, v in accuracy_concept_dict.items()]))
        res = pd.DataFrame()
        res["model"] = [model.__class__.__name__]
        res["clean_coverage_task"] = [coverage_task.item()]
        res["acc_test_task"] = [acc_test_task]
        res["clean_coverage_concepts"] = [coverage_concepts.item()]
        res["acc_test_concepts"] = [acc_test_concepts]
        for k, v in coverage_concept_dict.items():
            res["cov_" + k] = [v.item()]
            res["acc_" + k] = [accuracy_concept_dict[k]]
        res["loss"] = [model.psi_loss]
        # The ablation arms differ only in these two, and lambda is otherwise
        # recoverable only by string-parsing the model name.
        res["loss_tasks"] = [model.psi_loss_tasks]
        res["loss_concepts"] = [model.psi_loss_concepts]
        res["deferral_cost"] = [model.deferral_cost]
        res["label_sm"] = [model.label_smoothing_free]
        res["seed"] = [seed]
        res["training_mode"] = [model.training_mode]
        res["acc_task_expert"] = [
            accuracy_score(y_true.view(-1, model.n_tasks), y_preds_expert.view(-1, model.n_tasks))]
        res.to_csv(path_results, mode="a", index=False,
                   header=not os.path.exists(path_results))
        if also_scores:
            return res, y_true, y_preds, c_true, c_preds, c_logits, y_logits, y_preds_expert
        else:
            return res