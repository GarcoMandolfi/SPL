"""
Loader for the homebrew hierarchical-animals dataset.

Expected directory layout under `root_dir/animals/`:
    images/<image_id>.jpg
    concepts.csv       columns: image_id, <concept_1>, <concept_2>, ...   (binary 0/1)
    labels.csv         columns: image_id, class                            (string class name)
    splits.csv         columns: image_id, split                            ("train"/"val"/"test")
    hierarchy.csv      columns: parent, child                              (not used at training time)
    concept_names.csv  optional; not required by this loader
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transform(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class AnimalsDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        concept_names: list,
        class_to_idx: dict,
        transform=None,
        splits: Optional[pd.DataFrame] = None,
        expert: Optional[dict] = None,
    ):
        self.root = root
        self.split = split
        self.concept_names = concept_names
        self.class_to_idx = class_to_idx
        self.transform = transform
        # {image_id: (c_expert array, y_expert int)}; None -> oracle
        self.expert = expert

        if splits is None:
            splits = pd.read_csv(os.path.join(root, "splits.csv"))
        concepts = pd.read_csv(os.path.join(root, "concepts.csv"))
        labels = pd.read_csv(os.path.join(root, "labels.csv"))

        ids = splits.loc[splits["split"] == split, "image_id"].tolist()
        self.concepts = concepts[concepts["image_id"].isin(ids)].reset_index(drop=True)
        self.labels = labels.set_index("image_id").loc[self.concepts["image_id"]].reset_index()

        self.n_concepts = len(concept_names)
        self.n_classes = len(class_to_idx)

    def __len__(self):
        return len(self.concepts)

    def __getitem__(self, idx):
        row = self.concepts.iloc[idx]
        img_id = row["image_id"]

        img_path = os.path.join(self.root, "images", f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        c = torch.tensor(row[self.concept_names].values.astype(np.int64), dtype=torch.long)

        class_name = self.labels.iloc[idx]["class"]
        y = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)

        if self.expert is None:
            # Oracle expert: expert labels equal ground truth.
            y_expert = y.clone()
            c_expert = c.clone()
        else:
            c_exp, y_exp = self.expert[img_id]
            c_expert = torch.tensor(c_exp, dtype=torch.long)
            y_expert = torch.tensor(int(y_exp), dtype=torch.long)

        return img, y, y_expert, c, c_expert


def _make_expert_labels(concepts_df, labels_df, concept_names, class_to_idx,
                        config, seed):
    """Simulate an expert with a given competence, deterministically per image.

    Labels are drawn ONCE for the whole dataset rather than inside __getitem__:
    an expert that answers differently every epoch carries no input-dependent
    signal, so the deferral head could only learn the marginal error rate.

    expert_concept_accuracy / expert_task_accuracy of 1.0 reproduces the current
    oracle behaviour exactly. Below 1.0, mask_wrong starts firing and the l_2
    term of the DCBM loss stops being identically zero.
    """
    c_true = concepts_df[concept_names].to_numpy().astype(np.int64)
    y_true = np.array([class_to_idx[c] for c in labels_df["class"]], dtype=np.int64)

    c_acc = float(config.get("expert_concept_accuracy", 1.0))
    # Task expert defaults to an oracle, matching the CUB protocol in the paper.
    y_acc = float(config.get("expert_task_accuracy", 1.0))
    if c_acc >= 1.0 and y_acc >= 1.0:
        return c_true.copy(), y_true.copy()

    rng = np.random.default_rng(seed)

    flip = rng.random(c_true.shape) > c_acc
    c_expert = np.where(flip, 1 - c_true, c_true)

    n_cls = len(class_to_idx)
    wrong = rng.random(len(y_true)) > y_acc
    # offset in [1, n_cls-1] guarantees a genuinely different class
    offset = rng.integers(1, max(2, n_cls), size=len(y_true))
    y_expert = np.where(wrong, (y_true + offset) % n_cls, y_true)

    return c_expert, y_expert


def _compute_imbalance(concepts_df: pd.DataFrame, concept_names: list) -> np.ndarray:
    """Per-concept negative/positive ratio on the training split, clipped for stability."""
    mat = concepts_df[concept_names].values.astype(np.float64)
    n = mat.shape[0]
    pos = mat.sum(axis=0)
    neg = n - pos
    # Avoid div-by-zero and extreme weights
    pos = np.clip(pos, 1.0, None)
    ratio = neg / pos
    return np.clip(ratio, 0.1, 50.0)


def _stratified_split(labels_df: pd.DataFrame, seed: int,
                      fracs=(0.6, 0.2, 0.2)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cls, grp in labels_df.groupby("class"):
        ids = grp["image_id"].to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, int(round(n * fracs[1]))) if n >= 3 else (1 if n >= 2 else 0)
        n_test = max(1, int(round(n * fracs[2]))) if n >= 3 else 0
        # Never starve the training split.
        while n - n_val - n_test < 1 and (n_val + n_test) > 0:
            if n_test >= n_val:
                n_test -= 1
            else:
                n_val -= 1
        assignment = (["val"] * n_val + ["test"] * n_test
                      + ["train"] * (n - n_val - n_test))
        rows.extend(zip(ids, assignment))
    return pd.DataFrame(rows, columns=["image_id", "split"])


def generate_data(root_dir: str, config: dict, attr_dir: Optional[str] = None,
                  seed: Optional[int] = None):
    """
    Build train/val/test DataLoaders matching the CUB loader interface.

    Config keys:
        batch_size: int
        num_workers: int
        weight_loss: bool     -> if True, return per-concept imbalance; else None
        expert_concept_accuracy: float, default 1.0 (oracle)
        expert_task_accuracy:    float, default 1.0 (oracle)
        expert_seed:             int, default 0 -- expert labels are fixed per
                                 image, independent of the train/val/test seed
        (human_oracle ignored — superseded by the expert_* keys)

    seed: None  -> use the canonical splits.csv (all folds see the same data,
                   so fold variance measures initialisation only)
          int   -> resample a class-stratified split, giving folds real
                   sampling variance
    """
    root = os.path.join(root_dir, "animals")

    # Discover concepts and classes from the CSVs so `n_concepts`/`n_classes`
    # don't need to be hard-coded in the training script.
    concepts_df = pd.read_csv(os.path.join(root, "concepts.csv"))
    labels_df = pd.read_csv(os.path.join(root, "labels.csv"))
    concept_names = [c for c in concepts_df.columns if c != "image_id"]
    class_names = sorted(labels_df["class"].unique().tolist())
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    splits = None if seed is None else _stratified_split(labels_df, seed=seed)

    # Expert labels are drawn over the FULL dataset before splitting, so a given
    # image gets the same expert answer whichever split it lands in.
    aligned = labels_df.set_index("image_id").loc[concepts_df["image_id"]].reset_index()
    c_exp, y_exp = _make_expert_labels(
        concepts_df, aligned, concept_names, class_to_idx,
        config, seed=config.get("expert_seed", 0),
    )
    expert = {img_id: (c_exp[i], y_exp[i])
              for i, img_id in enumerate(concepts_df["image_id"])}

    train_ds = AnimalsDataset(root, "train", concept_names, class_to_idx,
                              _build_transform(train=True), splits=splits, expert=expert)
    val_ds = AnimalsDataset(root, "val", concept_names, class_to_idx,
                            _build_transform(train=False), splits=splits, expert=expert)
    test_ds = AnimalsDataset(root, "test", concept_names, class_to_idx,
                             _build_transform(train=False), splits=splits, expert=expert)

    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if config.get("weight_loss", False):
        imbalance = _compute_imbalance(train_ds.concepts, concept_names)
    else:
        imbalance = None

    return train_loader, val_loader, test_loader, imbalance
