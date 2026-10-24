"""Evaluate a checkpoint on the validation split and write a metrics bundle.

Outputs (in ``--output-dir``):
    metrics.json      accuracy, top-3/5, macro F1, per-class accuracy
    confusion.json    102x102 counts plus the strongest off-diagonal pairs
    predictions.csv   one row per image

``datasets/classes.json`` is written on the first run so that inference, this
report and the confusion-matrix figure all agree on what class index 0 means.
"""

import argparse
import csv
import json
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from .config import (
    DEFAULT_CHECKPOINTS,
    NUM_CLASSES,
    NUM_WORKERS,
    OUTPUT_DIR,
    VALID_DIR,
    default_checkpoint,
    load_class_names,
    save_class_order,
)
from .data import eval_transform, limit
from .engine import evaluate
from .metrics import classification_report, confusion_matrix, per_class_accuracy
from .model import load_checkpoint


def label_names(classes, names):
    """Index -> human-readable name, using the folder order of ``classes``."""
    return [names.get(c, f"class {c}") for c in classes]


def top_confusions(cm, labels, rows=10):
    """The off-diagonal cells that absorb the most mistakes."""
    pairs = []
    for true_idx, row in enumerate(cm):
        for pred_idx, count in enumerate(row):
            if true_idx != pred_idx and count:
                pairs.append({"true": labels[true_idx],
                              "predicted": labels[pred_idx],
                              "count": count})
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:rows]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None,
                        help="path to .pt/.pth (default: " +
                             ", ".join(os.path.basename(p) for p in DEFAULT_CHECKPOINTS) + ")")
    parser.add_argument("--data-dir", default=None,
                        help="folder with class sub-folders (default: datasets/flowers/valid)")
    parser.add_argument("--output-dir", default=os.path.join(OUTPUT_DIR, "eval"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="debug: only N images")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = default_checkpoint(args.checkpoint)
    print(f"checkpoint {checkpoint}\ndevice     {device}")

    root = args.data_dir or VALID_DIR
    base = datasets.ImageFolder(root, eval_transform(args.image_size))
    classes = base.classes
    save_class_order(classes)

    dataset = limit(base, args.limit) if args.limit else base
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    print(f"images {len(dataset)} | classes {len(classes)}")

    model, saved_best = load_checkpoint(checkpoint, NUM_CLASSES, device)
    if saved_best is not None:
        print(f"checkpoint reports best_acc {saved_best:.4f}")

    started = time.time()
    stats, y_true, y_pred, _ = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    elapsed = time.time() - started

    report = classification_report(y_true, y_pred, len(classes))
    cm = confusion_matrix(y_true, y_pred, len(classes))
    names = label_names(classes, load_class_names())
    per_class = per_class_accuracy(cm)
    per_class_named = {names[k]: round(v, 4) for k, v in per_class.items()}

    sampled = list(getattr(dataset, "indices", range(len(base.samples))))
    rows = [base.samples[i] for i in sampled]

    os.makedirs(args.output_dir, exist_ok=True)

    metrics = {
        "checkpoint": os.path.relpath(checkpoint),
        "split": os.path.relpath(root),
        "num_images": len(dataset),
        "num_classes": len(classes),
        "image_size": args.image_size,
        "inference_seconds": round(elapsed, 1),
        "images_per_second": round(len(dataset) / elapsed, 1),
        "loss": round(stats["loss"], 4),
        "accuracy": round(report["accuracy"], 4),
        "top3_accuracy": round(stats["top3"], 4),
        "top5_accuracy": round(stats["top5"], 4),
        "macro_precision": round(report["macro_precision"], 4),
        "macro_recall": round(report["macro_recall"], 4),
        "macro_f1": round(report["macro_f1"], 4),
        "weighted_f1": round(report["weighted_f1"], 4),
        "per_class_accuracy": per_class_named,
        "hardest_classes": sorted(
            ({"name": k, "accuracy": v} for k, v in per_class_named.items()),
            key=lambda item: item["accuracy"])[:8],
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=1)

    with open(os.path.join(args.output_dir, "confusion.json"), "w", encoding="utf-8") as fh:
        json.dump({"matrix": cm, "labels": names,
                   "top_confusions": top_confusions(cm, names)},
                  fh, ensure_ascii=False, indent=1)

    with open(os.path.join(args.output_dir, "predictions.csv"), "w",
              encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "true_index", "predicted_index", "correct"])
        for (path, true_index), pred_index in zip(rows, y_pred):
            writer.writerow([os.path.relpath(path), true_index, pred_index,
                             int(true_index == pred_index)])

    print(f"\naccuracy      {metrics['accuracy']:.4f}")
    print(f"top-3 / top-5 {metrics['top3_accuracy']:.4f} / {metrics['top5_accuracy']:.4f}")
    print(f"macro F1      {metrics['macro_f1']:.4f}")
    print(f"throughput    {metrics['images_per_second']:.0f} images/s "
          f"({elapsed:.1f}s for {len(dataset)})")
    print(f"\nwrote {args.output_dir}/metrics.json, confusion.json, predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
