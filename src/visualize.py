"""Figures for the README: training curves, confusion heatmap, prediction grid.

    python -m src.visualize --history outputs/resnet18-64/history.csv \
                            --metrics outputs/eval/metrics.json \
                            --confusion outputs/eval/confusion.json \
                            --checkpoint weights/flower_resnet18.pth
"""

import argparse
import csv
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from PIL import Image

from .config import (
    DOCS_IMAGES,
    INPUT_SIZE,
    NUM_CLASSES,
    VALID_DIR,
    load_class_names,
    load_class_order,
)
from .data import eval_transform
from .model import load_checkpoint

SURFACE = "#ffffff"
INK = "#1f2430"
MUTED = "#7a8194"
ACCENT = "#4c6ef5"
ACCENT_2 = "#12b886"
GRID = "#e6e9f0"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def read_history(path):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {
        "epoch": [int(r["epoch"]) for r in rows],
        "stage": [int(r.get("stage") or 0) for r in rows],
        "train_acc": [float(r["train_acc"]) for r in rows],
        "val_acc": [float(r["val_acc"]) for r in rows],
        "train_loss": [float(r["train_loss"]) for r in rows],
        "val_loss": [float(r["val_loss"]) for r in rows],
        "lr": [float(r["learning_rate"]) for r in rows],
    }


def _stage_boundary(data):
    """Epoch index where the backbone was unfrozen, or ``None``."""
    stages = data.get("stage") or []
    for i in range(1, len(stages)):
        if stages[i] != stages[i - 1]:
            return i
    return None


def _mark_stage(ax, data):
    boundary = _stage_boundary(data)
    if boundary is None:
        return
    ax.axvline(data["epoch"][boundary], color="#adb5bd", linestyle=":", linewidth=1)
    ax.text(data["epoch"][boundary], ax.get_ylim()[1], " unfrozen",
            color="#868e96", fontsize=8, va="top", ha="left")


def plot_training(history_path, output_path):
    data = read_history(history_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)

    ax = axes[0]
    _style(ax)
    ax.plot(data["epoch"], data["train_acc"], color=ACCENT, lw=2, label="train")
    ax.plot(data["epoch"], data["val_acc"], color=ACCENT_2, lw=2, label="validation")
    best_epoch = int(np.argmax(data["val_acc"]))
    ax.scatter([data["epoch"][best_epoch]], [data["val_acc"][best_epoch]],
               color=ACCENT_2, zorder=5, s=36)
    ax.annotate(f"best {data['val_acc'][best_epoch]:.3f}",
                (data["epoch"][best_epoch], data["val_acc"][best_epoch]),
                textcoords="offset points", xytext=(8, -14), color=INK, fontsize=9)
    ax.set_title("Accuracy per epoch", color=INK, fontsize=11)
    ax.set_xlabel("epoch", color=MUTED)
    _mark_stage(ax, data)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax = axes[1]
    _style(ax)
    ax.plot(data["epoch"], data["train_loss"], color=ACCENT, lw=2, label="train")
    ax.plot(data["epoch"], data["val_loss"], color="#f76707", lw=2, label="validation")
    ax.set_title("Cross-entropy loss", color=INK, fontsize=11)
    ax.set_xlabel("epoch", color=MUTED)
    _mark_stage(ax, data)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    ax = axes[2]
    _style(ax)
    ax.plot(data["epoch"], data["lr"], color=MUTED, lw=2)
    ax.set_yscale("log")
    ax.set_title("Learning rate schedule", color=INK, fontsize=11)
    ax.set_xlabel("epoch", color=MUTED)
    _mark_stage(ax, data)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"figure -> {output_path}")


def plot_confusion(confusion_path, output_path, limit=102):
    with open(confusion_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    matrix = np.array(payload["matrix"], dtype=float)
    labels = payload["labels"]

    # Every class has ground truth in this split, so the full matrix is shown;
    # tick labels are thinned out to stay legible.
    upto = min(len(labels), limit)
    sub = matrix[:upto, :upto]

    fig, ax = plt.subplots(figsize=(12.5, 11), facecolor=SURFACE)
    image = ax.imshow(np.where(sub > 0, sub, np.nan), cmap="Blues",
                      norm=LogNorm(vmin=1, vmax=max(sub.max(), 2)))
    diagonal = np.trace(sub) / sub.sum() if sub.sum() else 0
    ax.set_title(f"Confusion matrix — {upto} classes, log colour scale "
                 f"· correct {diagonal:.1%} of {int(sub.sum())} images",
                 color=INK, fontsize=12)
    step = max(1, upto // 26)
    ax.set_xticks(range(0, upto, step))
    ax.set_yticks(range(0, upto, step))
    ax.set_xticklabels([labels[i][:16] for i in range(0, upto, step)],
                       rotation=90, fontsize=7, color=MUTED)
    ax.set_yticklabels([labels[i][:16] for i in range(0, upto, step)],
                       fontsize=7, color=MUTED)
    ax.set_xlabel("predicted", color=MUTED)
    ax.set_ylabel("true", color=MUTED)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="images")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"figure -> {output_path}")


def plot_predictions(checkpoint, output_path, count=8, seed=7, root=VALID_DIR,
                     image_size=INPUT_SIZE):
    from torchvision import datasets

    names = load_class_names()
    labels = [names.get(folder, f"class {folder}") for folder in load_class_order()]
    model, _ = load_checkpoint(checkpoint, NUM_CLASSES)
    device = next(model.parameters()).device
    dataset = datasets.ImageFolder(root, eval_transform(image_size))
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), count)

    cols, rows = 4, (count + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows), facecolor=SURFACE)

    with torch.inference_mode():
        for ax, index in zip(axes.ravel(), indices):
            path, label = dataset.samples[index]
            image = Image.open(path).convert("RGB")
            tensor = eval_transform(image_size)(image).unsqueeze(0).to(device)
            probs = torch.softmax(model(tensor).float(), dim=1)[0]
            top_prob, top_index = probs.max(dim=0)
            predicted = int(top_index)

            ax.imshow(image)
            ax.axis("off")
            correct = predicted == label
            colour = ACCENT_2 if correct else "#e03131"
            ax.set_title(f"{labels[predicted]}  {top_prob:.2f}",
                         color=colour, fontsize=10, pad=6)
            if not correct:
                ax.text(0.5, -0.06, f"true: {labels[label]}",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=9, color=MUTED)

    fig.suptitle("Predictions on the validation split (green = correct)",
                 color=INK, fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {output_path}")


def plot_class_distribution(root, output_path):
    from torchvision import datasets
    from collections import Counter

    dataset = datasets.ImageFolder(root)
    counts = Counter(dataset.targets)
    values = [counts.get(i, 0) for i in range(len(dataset.classes))]
    names = load_class_names()

    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(14, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.bar(range(len(values)), [values[i] for i in order], color=ACCENT, width=0.8)
    step = 8
    ax.set_xticks(range(0, len(values), step))
    ax.set_xticklabels([names.get(str(order[i] + 1), "")[:14]
                        for i in range(0, len(values), step)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("training images", color=MUTED)
    ax.set_title(f"Class balance — {sum(values)} images over {len(values)} classes "
                 f"(min {min(values)}, max {max(values)})",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {output_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", default="outputs/resnet18-64/history.csv")
    parser.add_argument("--metrics", default="outputs/eval/metrics.json")
    parser.add_argument("--confusion", default="outputs/eval/confusion.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--train-dir", default="datasets/flowers/train")
    parser.add_argument("--valid-dir", default=VALID_DIR)
    parser.add_argument("--output-dir", default=DOCS_IMAGES)
    parser.add_argument("--only", default="all",
                        choices=["all", "training", "confusion", "predictions", "distribution"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    wanted = args.only

    if wanted in ("all", "training") and os.path.exists(args.history):
        plot_training(args.history, os.path.join(args.output_dir, "training_curves.png"))
    if wanted in ("all", "confusion") and os.path.exists(args.confusion):
        plot_confusion(args.confusion, os.path.join(args.output_dir, "confusion_matrix.png"))
    if wanted in ("all", "predictions") and args.checkpoint:
        plot_predictions(args.checkpoint,
                         os.path.join(args.output_dir, "sample_predictions.png"),
                         root=args.valid_dir)
    if wanted in ("all", "distribution") and os.path.isdir(args.train_dir):
        plot_class_distribution(args.train_dir,
                                os.path.join(args.output_dir, "class_distribution.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
