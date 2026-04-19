"""
dataset.py — ASL dataset loading, preprocessing, and augmentation.
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import (
    Dataset, DataLoader, ConcatDataset, Subset, WeightedRandomSampler
)
from torchvision import transforms

# Auto-detect best num_workers for the environment
def _get_num_workers():
    """Use 4 workers on Colab/Linux, 0 on Windows (avoids spawn issues)."""
    import platform
    if platform.system() == "Windows":
        return 0
    return 4


# ─── Constants ───────────────────────────────────────────────────────────────

NUM_CLASSES = 26           # A-Z all 26 letters
IMG_SIZE    = 64
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

# All 26 letters A-Z mapped to indices 0-25
LABEL_TO_LETTER = {i: chr(65 + i) for i in range(26)}
LETTER_TO_LABEL = {v: k for k, v in LABEL_TO_LETTER.items()}


# ─── Transforms ──────────────────────────────────────────────────────────────

def get_skeleton_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 10, IMG_SIZE + 10)),
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=25),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.80, 1.20),
            shear=10,
        ),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
    ])


def get_val_transforms():
    """Clean transforms for validation and test — no augmentation."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ─── Dataset Class ───────────────────────────────────────────────────────────

class ASLFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.samples   = []

        loaded = []
        for i in range(NUM_CLASSES):
            letter    = LABEL_TO_LETTER[i]
            class_dir = Path(root_dir) / letter
            if not class_dir.exists():
                continue
            found = (
                list(class_dir.glob("*.jpg")) +
                list(class_dir.glob("*.jpeg")) +
                list(class_dir.glob("*.png"))
            )
            for img_path in found:
                self.samples.append((str(img_path), i))
            if found:
                loaded.append(f"{letter}({len(found)})")

        if not self.samples:
            raise FileNotFoundError(
                f"No images found in {root_dir}. "
                "Expected subfolders named A-Z."
            )
        print(f"  Loaded {len(self.samples)} images — {', '.join(loaded)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_labels(self):
        return [label for _, label in self.samples]


# ─── Helper: split a dataset into three Subsets ──────────────────────────────

def _split_dataset(base_ds_train, base_ds_val, base_ds_test,
                   splits=(0.70, 0.15, 0.15), seed=42):
   
    n       = len(base_ds_train)
    n_train = int(n * splits[0])
    n_val   = int(n * splits[1])
    # n_test  = remainder

    indices = torch.randperm(
        n, generator=torch.Generator().manual_seed(seed)
    ).tolist()

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    return (
        Subset(base_ds_train, train_idx),
        Subset(base_ds_val,   val_idx),
        Subset(base_ds_test,  test_idx),
    )


# ─── Weighted Sampler ────────────────────────────────────────────────────────

def _get_labels_from_subset(subset):
    ds = subset.dataset
    return [ds.samples[i][1] for i in subset.indices]


def make_weighted_sampler(concat_train: ConcatDataset) -> WeightedRandomSampler:
  
    all_labels = []
    for ds in concat_train.datasets:
        if isinstance(ds, Subset):
            all_labels.extend(_get_labels_from_subset(ds))
        elif isinstance(ds, ASLFolderDataset):
            all_labels.extend(ds.get_labels())

    class_counts   = Counter(all_labels)
    sample_weights = torch.tensor(
        [1.0 / class_counts[label] for label in all_labels],
        dtype=torch.float
    )

    print("\n  Class counts (training combined):")
    for idx in sorted(class_counts):
        bar = "█" * (class_counts[idx] // 20)
        print(f"    {LABEL_TO_LETTER[idx]}: {bar:<40} {class_counts[idx]}")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(all_labels),
        replacement=True,
    )


# ─── Data Loaders ────────────────────────────────────────────────────────────

def get_folder_loaders(data_dir: str, batch_size: int = 32,
                       splits=(0.70, 0.15, 0.15)):
    """Single folder → train/val/test split."""
    train_ds = ASLFolderDataset(data_dir, transform=get_skeleton_train_transforms())
    val_ds   = ASLFolderDataset(data_dir, transform=get_val_transforms())
    test_ds  = ASLFolderDataset(data_dir, transform=get_val_transforms())

    train_sub, val_sub, test_sub = _split_dataset(
        train_ds, val_ds, test_ds, splits=splits
    )

    n_train = len(train_sub)
    n_val   = len(val_sub)
    n_test  = len(test_sub)
    print(f"Dataset  →  train: {n_train}  val: {n_val}  test: {n_test}")

    nw = _get_num_workers()
    pm = nw > 0
    return (
        DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                   num_workers=nw, pin_memory=pm, persistent_workers=pm),
        DataLoader(val_sub,   batch_size=batch_size, shuffle=False,
                   num_workers=nw, pin_memory=pm, persistent_workers=pm),
        DataLoader(test_sub,  batch_size=batch_size, shuffle=False,
                   num_workers=nw, pin_memory=pm, persistent_workers=pm),
    )


def get_combined_loaders(
    custom_dir: str,
    extra_dir: str = None,
    batch_size: int = 32,
    splits: tuple = (0.70, 0.15, 0.15),
    use_weighted_sampler: bool = True,
):
    
    print(f"\n── Custom dataset ({custom_dir}) ──")
    c_train_ds = ASLFolderDataset(custom_dir, transform=get_skeleton_train_transforms())
    c_val_ds   = ASLFolderDataset(custom_dir, transform=get_val_transforms())
    c_test_ds  = ASLFolderDataset(custom_dir, transform=get_val_transforms())

    c_train, c_val, c_test = _split_dataset(c_train_ds, c_val_ds, c_test_ds, splits)
    print(f"  → train: {len(c_train)}  val: {len(c_val)}  test: {len(c_test)}")

    e_train, e_val, e_test = None, None, None
    if extra_dir and os.path.exists(extra_dir):
        try:
            print(f"\n── Extra dataset ({extra_dir}) ──")
            ex_train_ds = ASLFolderDataset(extra_dir, transform=get_skeleton_train_transforms())
            ex_val_ds   = ASLFolderDataset(extra_dir, transform=get_val_transforms())
            ex_test_ds  = ASLFolderDataset(extra_dir, transform=get_val_transforms())

            e_train, e_val, e_test = _split_dataset(
                ex_train_ds, ex_val_ds, ex_test_ds, splits
            )
            print(f"  → train: {len(e_train)}  val: {len(e_val)}  test: {len(e_test)}")
        except FileNotFoundError:
            print(f"  Extra dir has no valid images — skipping")

    # ── Combine each split ────────────────────────────────────────────────
    train_sources = [s for s in [c_train, e_train] if s is not None]
    val_sources   = [s for s in [c_val,   e_val]   if s is not None]
    test_sources  = [s for s in [c_test,  e_test]  if s is not None]

    combined_train = ConcatDataset(train_sources)
    combined_val   = ConcatDataset(val_sources)
    combined_test  = ConcatDataset(test_sources)

    print(f"\n── Combined totals ──")
    print(f"  train : {len(combined_train)}")
    print(f"  val   : {len(combined_val)}")
    print(f"  test  : {len(combined_test)}")

    nw = _get_num_workers()
    pm = nw > 0

    if use_weighted_sampler and e_train is not None:
        print("\nBuilding weighted sampler for training split...")
        sampler      = make_weighted_sampler(combined_train)
        train_loader = DataLoader(combined_train, batch_size=batch_size,
                                  sampler=sampler, num_workers=nw,
                                  pin_memory=pm, persistent_workers=pm)
    else:
        train_loader = DataLoader(combined_train, batch_size=batch_size,
                                  shuffle=True, num_workers=nw,
                                  pin_memory=pm, persistent_workers=pm)

    val_loader  = DataLoader(combined_val,  batch_size=batch_size, shuffle=False,
                             num_workers=nw, pin_memory=pm, persistent_workers=pm)
    test_loader = DataLoader(combined_test, batch_size=batch_size, shuffle=False,
                             num_workers=nw, pin_memory=pm, persistent_workers=pm)

    return train_loader, val_loader, test_loader


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    choices=["folder", "combined"], default="folder")
    parser.add_argument("--data_dir",  default="data/custom")
    parser.add_argument("--extra_dir", default=None)
    args = parser.parse_args()

    if args.source == "combined":
        train_loader, val_loader, test_loader = get_combined_loaders(
            custom_dir=args.data_dir, extra_dir=args.extra_dir
        )
    else:
        train_loader, val_loader, test_loader = get_folder_loaders(args.data_dir)

    imgs, labels = next(iter(train_loader))
    print(f"\nBatch   : {imgs.shape}")
    print(f"Labels  : {[LABEL_TO_LETTER[l.item()] for l in labels[:8]]}")