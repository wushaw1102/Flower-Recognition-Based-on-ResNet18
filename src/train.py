"""Two-stage fine-tuning of ResNet-18 on the 102-class flower dataset.

Stage 1 trains only the new classifier on frozen ImageNet features; stage 2
unfreezes the backbone and continues with a smaller learning rate. That is the
schedule the published checkpoint came from, and the reason training converges
in a few epochs instead of overfitting a randomly initialised ResNet.
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from .config import (
    BATCH_SIZE,
    INPUT_SIZE,
    NUM_CLASSES,
    NUM_WORKERS,
    OUTPUT_DIR,
    SEED,
    TRAIN_DIR,
    VALID_DIR,
)
from .data import build_datasets, build_loaders, limit
from .engine import History, fit
from .model import build_model, count_parameters, unfreeze


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_history(history: History, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                         "top3", "top5", "learning_rate", "seconds"])
        for row in history.rows:
            writer.writerow([row.epoch, f"{row.train_loss:.6f}", f"{row.train_acc:.6f}",
                             f"{row.val_loss:.6f}", f"{row.val_acc:.6f}",
                             f"{row.top3:.6f}", f"{row.top5:.6f}",
                             f"{row.lr:.10f}", f"{row.seconds:.1f}"])
    print(f"history -> {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", default=TRAIN_DIR)
    parser.add_argument("--valid-dir", default=VALID_DIR)
    parser.add_argument("--output-dir", default=os.path.join(OUTPUT_DIR, "resnet18-64"))
    parser.add_argument("--epochs", type=int, default=5, help="epochs per stage")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=INPUT_SIZE)
    parser.add_argument("--lr-stage1", type=float, default=1e-2)
    parser.add_argument("--lr-stage2", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    parser.add_argument("--patience", type=int, default=0, help="early stopping patience")
    parser.add_argument("--plateau", action="store_true",
                        help="ReduceLROnPlateau instead of StepLR")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="skip ImageNet weights (offline debugging)")
    parser.add_argument("--limit-train", type=int, default=0, help="debug: cap train set")
    parser.add_argument("--limit-valid", type=int, default=0, help="debug: cap valid set")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} | torch {torch.__version__} | image {args.image_size}px")

    train_ds, valid_ds = build_datasets(args.train_dir, args.valid_dir, args.image_size)
    train_ds = limit(train_ds, args.limit_train)
    valid_ds = limit(valid_ds, args.limit_valid)
    train_loader, valid_loader = build_loaders(
        train_ds, valid_ds, args.batch_size, args.workers)
    print(f"train {len(train_ds)} images | valid {len(valid_ds)} images "
          f"| {len(train_ds.dataset.classes)} classes")

    model = build_model(NUM_CLASSES, pretrained=not args.no_pretrained,
                        freeze_backbone=True).to(device)
    print(f"parameters {count_parameters(model) / 1e6:.1f} M "
          f"(stage 1 trains the classifier only)")

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_path = os.path.join(args.output_dir, "best.pt")
    history = History()

    # ---- stage 1: feature extraction ------------------------------------- #
    print("\n=== stage 1 / frozen backbone ===")
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr_stage1)
    scheduler = (ReduceLROnPlateau(optimizer, patience=2, factor=0.3)
                 if args.plateau else StepLR(optimizer, step_size=10, gamma=0.1))
    fit(model, train_loader, valid_loader, criterion, optimizer, device,
        args.epochs, scheduler=scheduler, scaler=scaler, patience=args.patience,
        checkpoint_path=best_path, history=history)

    # ---- stage 2: fine-tune everything ----------------------------------- #
    print("\n=== stage 2 / full fine-tuning ===")
    unfreeze(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_stage2)
    scheduler = (ReduceLROnPlateau(optimizer, patience=2, factor=0.3)
                 if args.plateau else StepLR(optimizer, step_size=7, gamma=0.1))
    fit(model, train_loader, valid_loader, criterion, optimizer, device,
        args.epochs, scheduler=scheduler, scaler=scaler, patience=args.patience,
        checkpoint_path=best_path, history=history)

    save_history(history, os.path.join(args.output_dir, "history.csv"))
    print(f"\nbest validation accuracy {history.best_val_acc:.4f}")
    print(f"checkpoint -> {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
