"""Convert the raw training logs written by ``legacy/fourth.py`` into the
history format this package uses.

The published checkpoint was produced by that script, and it wrote one CSV per
stage with a different set of columns. Rather than pretend the current trainer
produced those numbers, this script restates them:

    python scripts/import_training_log.py \
        --stage1 legacy/training_results_stage1.csv \
        --stage2 legacy/training_results_stage2.csv \
        --output outputs/resnet18-64/history.csv

Stage 2 epochs are renumbered to continue after stage 1 so the curves are
continuous, and a ``stage`` column records where the backbone was unfrozen.
"""

import argparse
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_stage(path, stage, offset):
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "epoch": int(row["epoch"]) + offset,
                "stage": stage,
                "train_loss": row["train_loss"],
                "train_acc": row["train_acc"],
                "val_loss": row["val_loss"],
                "val_acc": row["val_acc"],
                "learning_rate": row["learning_rate"],
            })
    return rows, offset + len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", default=os.path.join(ROOT, "legacy", "training_results_stage1.csv"))
    parser.add_argument("--stage2", default=os.path.join(ROOT, "legacy", "training_results_stage2.csv"))
    parser.add_argument("--output",
                        default=os.path.join(ROOT, "outputs", "resnet18-64", "history.csv"))
    args = parser.parse_args()

    rows, offset = read_stage(args.stage1, 1, 0)
    more, _ = read_stage(args.stage2, 2, offset)
    rows += more

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "stage", "train_loss", "train_acc",
                                                "val_loss", "val_acc", "learning_rate"])
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda r: float(r["val_acc"]))
    print(f"{len(rows)} epochs ({offset} in stage 1) -> {args.output}")
    print(f"best val_acc {float(best['val_acc']):.4f} at epoch {best['epoch']} "
          f"(stage {best['stage']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
