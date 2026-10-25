"""
Prepare the per-image CSV files that animals_loader.py expects.

Usage:
    python prepare_dataset.py --data_dir data/animals

Prerequisites:
    1.  Place your images in  data/animals/images/
        Named as  <class>_<anything>.jpg   (case-insensitive class prefix)
        Examples: eagle_01.jpg, Frog_photo3.jpg, SALMON_side.jpg

    2.  Keep class_concepts.csv  (the per-class reference we built earlier)
        in the same data/animals/ directory.

The script produces three files the loader needs:
    data/animals/concepts.csv   – image_id, <concept_1>, ..., <concept_n>
    data/animals/labels.csv     – image_id, class
    data/animals/splits.csv     – image_id, split   (train / val / test)

Split strategy (configurable via --train/--val/--test fractions):
    Default 60 / 20 / 20, stratified by class so every class appears in
    every split when there are enough images.
"""

import argparse
import os
import random
from collections import defaultdict

import pandas as pd


def infer_class(filename: str, known_classes: list[str]) -> str | None:
    """Return the class whose name is a prefix of `filename` (case-insensitive)."""
    name = filename.lower().split(".")[0]          # strip extension
    for cls in sorted(known_classes, key=len, reverse=True):  # longest match first
        if name.startswith(cls.lower()):
            return cls
    return None


def stratified_split(ids_by_class: dict, train_frac: float, val_frac: float, seed: int):
    """Assign each image to a split, keeping class proportions."""
    rng = random.Random(seed)
    assignments = {}
    for cls, ids in ids_by_class.items():
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, round(n * train_frac))
        n_val = max(1, round(n * val_frac)) if n > 1 else 0
        # Ensure at least 1 in test when possible
        if n_train + n_val >= n and n > 2:
            n_val = max(1, n - n_train - 1)
        for i, img_id in enumerate(ids):
            if i < n_train:
                assignments[img_id] = "train"
            elif i < n_train + n_val:
                assignments[img_id] = "val"
            else:
                assignments[img_id] = "test"
    return assignments


def main():
    parser = argparse.ArgumentParser(description="Generate per-image CSVs for the animals dataset.")
    parser.add_argument("--data_dir", type=str, default="data/animals",
                        help="Root of the animals dataset (contains images/ and class_concepts.csv)")
    parser.add_argument("--train", type=float, default=0.6)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.data_dir
    img_dir = os.path.join(root, "images")
    ref_path = os.path.join(root, "class_concepts.csv")

    # ---- Load the class-level reference ----
    ref = pd.read_csv(ref_path)
    known_classes = ref["class"].tolist()
    concept_cols = [c for c in ref.columns if c != "class"]
    class_to_concepts = {row["class"]: row[concept_cols].to_dict() for _, row in ref.iterrows()}

    # ---- Scan images and match to classes ----
    image_files = sorted(f for f in os.listdir(img_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not image_files:
        raise FileNotFoundError(f"No images found in {img_dir}")

    ids_by_class = defaultdict(list)
    rows_concepts = []
    rows_labels = []
    skipped = []

    for fname in image_files:
        cls = infer_class(fname, known_classes)
        if cls is None:
            skipped.append(fname)
            continue

        img_id = os.path.splitext(fname)[0]
        ids_by_class[cls].append(img_id)
        concept_vals = class_to_concepts[cls]
        rows_concepts.append({"image_id": img_id, **concept_vals})
        rows_labels.append({"image_id": img_id, "class": cls})

    if skipped:
        print(f"WARNING: {len(skipped)} images could not be matched to a class and were skipped:")
        for s in skipped:
            print(f"  {s}")

    # ---- Assign splits ----
    assignments = stratified_split(ids_by_class, args.train, args.val, args.seed)
    rows_splits = [{"image_id": img_id, "split": split} for img_id, split in assignments.items()]

    # ---- Write CSVs ----
    df_concepts = pd.DataFrame(rows_concepts)
    # Ensure column order: image_id first, then concept columns in reference order
    df_concepts = df_concepts[["image_id"] + concept_cols]
    df_concepts.to_csv(os.path.join(root, "concepts.csv"), index=False)

    df_labels = pd.DataFrame(rows_labels)
    df_labels.to_csv(os.path.join(root, "labels.csv"), index=False)

    df_splits = pd.DataFrame(rows_splits)
    df_splits.to_csv(os.path.join(root, "splits.csv"), index=False)

    # ---- Summary ----
    split_counts = df_splits["split"].value_counts()
    print(f"\nDataset prepared in {root}/")
    print(f"  Images matched:  {len(df_concepts)}")
    print(f"  Classes found:   {list(ids_by_class.keys())}")
    print(f"  Concepts:        {len(concept_cols)}")
    print(f"  Split sizes:     train={split_counts.get('train', 0)}  "
          f"val={split_counts.get('val', 0)}  test={split_counts.get('test', 0)}")


if __name__ == "__main__":
    main()
