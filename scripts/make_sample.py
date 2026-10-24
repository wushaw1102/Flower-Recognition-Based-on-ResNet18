"""Build a small, class-balanced sample of the dataset for the demo and for
running the evaluation pipeline without the 300 MB download.

    python scripts/make_sample.py --per-class 2

Copies ``--per-class`` validation images from every class into
``datasets/sample/images`` and writes ``datasets/sample/index.csv``, which the
Flask demo reads to populate its gallery.
"""

import argparse
import csv
import json
import os
import random
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALID_DIR = os.path.join(ROOT, "datasets", "flowers", "valid")
SAMPLE_DIR = os.path.join(ROOT, "datasets", "sample")
CAT_TO_NAME = os.path.join(ROOT, "datasets", "cat_to_name.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--source", default=VALID_DIR)
    parser.add_argument("--output", default=SAMPLE_DIR)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    with open(CAT_TO_NAME, encoding="utf-8") as fh:
        names = json.load(fh)

    image_dir = os.path.join(args.output, "images")
    os.makedirs(image_dir, exist_ok=True)
    rng = random.Random(args.seed)

    rows = []
    classes = sorted(
        d for d in os.listdir(args.source)
        if os.path.isdir(os.path.join(args.source, d))
    )
    for folder in classes:
        files = sorted(
            f for f in os.listdir(os.path.join(args.source, folder))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        for original in rng.sample(files, min(args.per_class, len(files))):
            target = f"class{folder}_{original}"
            shutil.copy2(os.path.join(args.source, folder, original),
                         os.path.join(image_dir, target))
            rows.append({
                "file_name": target,
                "class_folder": folder,
                "class_name": names.get(folder, folder),
            })

    index_path = os.path.join(args.output, "index.csv")
    with open(index_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file_name", "class_folder", "class_name"])
        writer.writeheader()
        writer.writerows(rows)

    total_mb = sum(os.path.getsize(os.path.join(image_dir, r["file_name"]))
                   for r in rows) / 1048576
    print(f"{len(rows)} images from {len(classes)} classes -> {image_dir} "
          f"({total_mb:.1f} MB)")
    print(f"index -> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
