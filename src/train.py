"""
train.py — Training loop for ASL recognition models.
"""

import os
import time
import argparse
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import (
    get_folder_loaders, get_combined_loaders, NUM_CLASSES
)
from custom_cnn import build_custom_cnn
from transfer_model import build_transfer_model, unfreeze_top_layers


# ─── Device ──────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slow — use Colab for GPU)")
    return device


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_accuracy(preds, targets):
    return (preds.argmax(dim=1) == targets).float().mean().item()


def compute_f1(preds, targets):
    return f1_score(
        targets.cpu().numpy(),
        preds.argmax(dim=1).cpu().numpy(),
        average="weighted", zero_division=0
    )


# ─── Training Step ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = total_acc = total_f1 = 0.0

    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_acc  += compute_accuracy(logits.detach(), labels)
        total_f1   += compute_f1(logits.detach(), labels)
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    n = len(loader)
    return total_loss / n, total_acc / n, total_f1 / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_acc = total_f1 = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item()
        total_acc  += compute_accuracy(logits, labels)
        total_f1   += compute_f1(logits, labels)
    n = len(loader)
    return total_loss / n, total_acc / n, total_f1 / n


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=8, delta=1e-4):
        self.patience   = patience
        self.delta      = delta
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def __call__(self, val_acc):
        if self.best_score is None or val_acc > self.best_score + self.delta:
            self.best_score = val_acc
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ─── Main Train Function ──────────────────────────────────────────────────────

def train(
    model, train_loader, val_loader, device,
    epochs=50, lr=1e-3, weight_decay=1e-4,
    save_path="models/model.pth", run_name="run",
    fine_tune_epoch=None, backbone=None,
):
    model = model.to(device)
    os.makedirs(Path(save_path).parent, exist_ok=True)

    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer  = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler     = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    early_stop = EarlyStopping(patience=10)

    try:
        writer = SummaryWriter(log_dir=f"runs/{run_name}")
        use_tb = True
    except Exception:
        writer = None
        use_tb = False

    best_val_acc = 0.0
    history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc","train_f1","val_f1"]}

    print(f"\n{'═'*60}")
    print(f"  Model    : {run_name}")
    print(f"  Epochs   : {epochs}  |  LR: {lr}  |  Device: {device}")
    print(f"  Classes  : {NUM_CLASSES}  (A-Z)")
    print(f"{'═'*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Fine-tune phase
        if fine_tune_epoch and backbone and epoch == fine_tune_epoch:
            print(f"\n  → Epoch {epoch}: Fine-tuning backbone (unfreezing top layers)")
            model     = unfreeze_top_layers(model, backbone, n=5)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr * 0.1, weight_decay=weight_decay
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch, eta_min=lr * 0.001
            )

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        if use_tb and writer:
            writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss},  epoch)
            writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},   epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        for k, v in [("train_loss", train_loss), ("val_loss", val_loss),
                     ("train_acc",  train_acc),  ("val_acc",  val_acc),
                     ("train_f1",   train_f1),   ("val_f1",   val_f1)]:
            history[k].append(v)

        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_f1":  val_f1,
            }, save_path)
            tag = "  ✓"

        print(
            f"  Epoch {epoch:3d}/{epochs}"
            f"  loss {train_loss:.3f}/{val_loss:.3f}"
            f"  acc {train_acc:.3f}/{val_acc:.3f}"
            f"  F1 {val_f1:.3f}"
            f"  {time.time()-t0:.0f}s{tag}"
        )

        early_stop(val_acc)
        if early_stop.stop:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    if use_tb and writer:
        writer.close()

    hist_path = save_path.replace(".pth", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Saved to          : {save_path}")
    return history


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ASL recognition (26 classes, skeleton data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
LOCAL:
  python src/train.py --model cnn --source combined \\
      --data_dir data/custom --extra_dir data/extra_skeleton \\
      --epochs 50 --save_path models/custom_cnn.pth

COLAB (mount Drive first):
  !python src/train.py --model transfer --backbone mobilenetv2 \\
      --source combined \\
      --data_dir /content/drive/MyDrive/Capstone/data/custom \\
      --extra_dir /content/drive/MyDrive/Capstone/data/extra_skeleton \\
      --epochs 60 --fine_tune_epoch 10 --batch_size 64 --lr 0.0005 \\
      --save_path /content/drive/MyDrive/Capstone/models/mobilenetv2.pth
        """
    )
    parser.add_argument("--model",      choices=["cnn", "transfer"], required=True)
    parser.add_argument("--backbone",   choices=["mobilenetv2", "resnet50", "efficientnet", "vgg16"],
                        default="mobilenetv2")
    parser.add_argument("--cnn_variant",choices=["standard", "deep"], default="standard")
    parser.add_argument("--source",     choices=["folder", "combined"], default="folder")
    parser.add_argument("--data_dir",   default="data/custom",
                        help="Custom skeleton dataset folder")
    parser.add_argument("--extra_dir",  default=None,
                        help="Extra ASL skeleton folder (optional)")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--save_path",  default="models/model.pth")
    parser.add_argument("--fine_tune_epoch", type=int, default=None)
    args = parser.parse_args()

    device = get_device()

    if args.source == "combined":
        train_loader, val_loader, _ = get_combined_loaders(
            custom_dir=args.data_dir,
            extra_dir=args.extra_dir,
            batch_size=args.batch_size,
        )
    else:
        train_loader, val_loader, _ = get_folder_loaders(
            args.data_dir, batch_size=args.batch_size
        )

    if args.model == "cnn":
        model    = build_custom_cnn(variant=args.cnn_variant, num_classes=NUM_CLASSES)
        run_name = f"custom_cnn_{args.cnn_variant}"
        backbone = None
    else:
        model    = build_transfer_model(backbone=args.backbone, num_classes=NUM_CLASSES,
                                        freeze_backbone=True)
        run_name = f"transfer_{args.backbone}"
        backbone = args.backbone

    train(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr,
        save_path=args.save_path, run_name=run_name,
        fine_tune_epoch=args.fine_tune_epoch,
        backbone=backbone,
    )