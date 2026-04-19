"""
evaluate.py — Full evaluation of trained ASL models.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm

from dataset import get_folder_loaders, get_combined_loaders, NUM_CLASSES, LABEL_TO_LETTER
from custom_cnn import build_custom_cnn
from transfer_model import build_transfer_model

# Stable list used everywhere — covers all 26 classes even if test split is missing some
ALL_LABELS      = list(range(NUM_CLASSES))
ALL_LABEL_NAMES = [LABEL_TO_LETTER[i] for i in ALL_LABELS]


# ─── Load Model ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, model_type: str, backbone: str = "mobilenetv2") -> nn.Module:
    if model_type == "cnn":
        model = build_custom_cnn(num_classes=NUM_CLASSES)
    else:
        model = build_transfer_model(backbone=backbone, num_classes=NUM_CLASSES,
                                     freeze_backbone=False)

    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    saved_acc = ckpt.get("val_acc", None)
    acc_str   = f"{saved_acc:.4f}" if saved_acc is not None else "?"
    print(f"Loaded {model_type} from {checkpoint_path}  (val_acc={acc_str})")
    return model


# ─── Inference ───────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: nn.Module, loader, device) -> tuple:
    """Returns (all_preds, all_targets, all_probs)."""
    model = model.to(device)
    model.eval()
    preds, targets, probs = [], [], []

    for imgs, labels in tqdm(loader, desc="  Evaluating", leave=False):
        imgs   = imgs.to(device)
        logits = model(imgs)
        prob   = torch.softmax(logits, dim=1)
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        targets.extend(labels.numpy())
        probs.extend(prob.cpu().numpy())

    return np.array(preds), np.array(targets), np.array(probs)


# ─── Metrics Summary ─────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, targets: np.ndarray,
                    model_name: str = "Model") -> dict:
    # Always pass labels= so sklearn never infers the class list from the data
    acc      = accuracy_score(targets, preds)
    prec     = precision_score(targets, preds, labels=ALL_LABELS,
                               average="weighted", zero_division=0)
    rec      = recall_score(targets, preds, labels=ALL_LABELS,
                            average="weighted", zero_division=0)
    f1_w     = f1_score(targets, preds, labels=ALL_LABELS,
                        average="weighted", zero_division=0)
    f1_macro = f1_score(targets, preds, labels=ALL_LABELS,
                        average="macro", zero_division=0)

    metrics = {
        "accuracy":    round(acc,      4),
        "precision":   round(prec,     4),
        "recall":      round(rec,      4),
        "f1_weighted": round(f1_w,     4),
        "f1_macro":    round(f1_macro, 4),
    }

    print(f"\n{'─'*40}")
    print(f"  Results — {model_name}")
    print(f"{'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")

    return metrics


# ─── Confusion Matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(preds, targets, title: str, save_path: str):
    # labels= ensures the matrix is always 26×26
    cm = confusion_matrix(targets, preds, labels=ALL_LABELS)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=ALL_LABEL_NAMES, yticklabels=ALL_LABEL_NAMES,
        linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title(title, fontsize=15, pad=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {save_path}")


# ─── Per-class F1 Chart ──────────────────────────────────────────────────────

def plot_per_class_f1(preds, targets, title: str, save_path: str):
    # labels= keeps the array length at exactly NUM_CLASSES
    f1_per_class = f1_score(targets, preds, labels=ALL_LABELS,
                            average=None, zero_division=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = [
        "#4C72B0" if f >= 0.90 else "#DD8452" if f >= 0.75 else "#C44E52"
        for f in f1_per_class
    ]
    bars = ax.bar(ALL_LABEL_NAMES, f1_per_class, color=colors,
                  width=0.65, edgecolor="white", linewidth=0.5)

    ax.set_ylim(0, 1.05)
    ax.axhline(0.90, color="gray", linewidth=0.8, linestyle="--",
               alpha=0.6, label="90% threshold")
    ax.set_xlabel("ASL Letter", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)

    for bar, f in zip(bars, f1_per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{f:.2f}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved per-class F1  → {save_path}")


# ─── Full Evaluation Pipeline ─────────────────────────────────────────────────

def evaluate_model(
    checkpoint_path: str,
    model_type: str,
    test_loader,
    device,
    results_dir: str = "results",
    backbone: str = "mobilenetv2",
    name: str = None,
) -> dict:
    import json as _json

    os.makedirs(results_dir, exist_ok=True)
    name = name or model_type

    model  = load_model(checkpoint_path, model_type, backbone)
    preds, targets, _ = run_inference(model, test_loader, device)

    metrics = compute_metrics(preds, targets, model_name=name)

    # Classification report — always uses all 26 labels
    report = classification_report(
        targets, preds,
        labels=ALL_LABELS,          # ← fix: explicit label list
        target_names=ALL_LABEL_NAMES,
        zero_division=0
    )
    report_path = os.path.join(results_dir, f"{name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report — {name}\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"  Saved report       → {report_path}")

    # Metrics JSON (for compare.py)
    metrics_path = os.path.join(results_dir, f"{name}_metrics.json")
    with open(metrics_path, "w") as f:
        _json.dump(metrics, f, indent=2)
    print(f"  Saved metrics JSON → {metrics_path}")

    # Plots
    plot_confusion_matrix(
        preds, targets,
        title=f"Confusion Matrix — {name}",
        save_path=os.path.join(results_dir, f"{name}_confusion_matrix.png")
    )
    plot_per_class_f1(
        preds, targets,
        title=f"Per-class F1-Score — {name}",
        save_path=os.path.join(results_dir, f"{name}_per_class_f1.png")
    )

    return metrics


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASL recognition models")
    parser.add_argument("--model_cnn",      default=None,
                        help="Path to custom CNN checkpoint")
    parser.add_argument("--model_transfer", default=None,
                        help="Path to transfer model checkpoint")
    parser.add_argument("--backbone",       default="mobilenetv2",
                        choices=["mobilenetv2", "resnet50", "efficientnet", "vgg16"])
    parser.add_argument("--data_dir",       default="data/custom",
                        help="Custom skeleton dataset folder (A-Z subfolders)")
    parser.add_argument("--extra_dir",      default=None,
                        help="Extra ASL skeleton folder (optional)")
    parser.add_argument("--source",         choices=["folder", "combined"],
                        default="folder")
    parser.add_argument("--results_dir",    default="results")
    parser.add_argument("--batch_size",     type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.source == "combined" and args.extra_dir:
        _, _, test_loader = get_combined_loaders(
            custom_dir=args.data_dir,
            extra_dir=args.extra_dir,
            batch_size=args.batch_size,
        )
    else:
        _, _, test_loader = get_folder_loaders(
            args.data_dir, batch_size=args.batch_size
        )

    results = {}

    if args.model_cnn:
        results["custom_cnn"] = evaluate_model(
            args.model_cnn, "cnn", test_loader, device,
            results_dir=args.results_dir, name="custom_cnn"
        )

    if args.model_transfer:
        results["transfer"] = evaluate_model(
            args.model_transfer, "transfer", test_loader, device,
            results_dir=args.results_dir, backbone=args.backbone,
            name=f"transfer_{args.backbone}"
        )

    if len(results) == 2:
        print(f"\n{'═'*55}")
        print(f"  {'Metric':<22} {'Custom CNN':>12} {'Transfer':>12}")
        print(f"{'─'*55}")
        for metric in ["accuracy", "precision", "recall", "f1_weighted", "f1_macro"]:
            cnn_v = results["custom_cnn"].get(metric, 0)
            trf_v = results["transfer"].get(metric, 0)
            winner = "  ←" if trf_v > cnn_v else ""
            print(f"  {metric:<22} {cnn_v:>12.4f} {trf_v:>12.4f}{winner}")
        print(f"{'═'*55}")