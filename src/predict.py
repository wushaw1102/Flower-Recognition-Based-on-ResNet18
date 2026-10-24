"""Run inference on one image or a whole folder.

    python -m src.predict --image datasets/sample/images/valid_0123_45.jpg
    python -m src.predict --dir ~/my_flowers --topk 3 --output outputs/preds.csv
"""

import argparse
import csv
import glob
import json
import os

import torch
from PIL import Image

from .config import (
    INPUT_SIZE,
    NUM_CLASSES,
    default_checkpoint,
    load_class_names,
    load_class_order,
)
from .data import eval_transform
from .model import load_checkpoint

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class Predictor:
    """Thin wrapper: image file in, ranked label probabilities out."""

    def __init__(self, checkpoint=None, device=None, image_size: int = INPUT_SIZE):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = default_checkpoint(checkpoint)
        self.model, _ = load_checkpoint(self.checkpoint, NUM_CLASSES, self.device)
        self.transform = eval_transform(image_size)

        names = load_class_names()
        # class index -> folder name -> human name. The folder order is *not*
        # numeric ('1', '10', '100', ...), so it is read from classes.json.
        self.labels = [names.get(folder, f"class {folder}")
                       for folder in load_class_order()]

    def predict(self, image, topk: int = 5):
        if isinstance(image, str):
            image = Image.open(image)
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits.float(), dim=1)[0]

        topk = min(topk, probs.numel())
        values, indices = probs.topk(topk)
        return [{"label": self.labels[int(i)], "class_index": int(i),
                 "probability": round(float(v), 4)}
                for v, i in zip(values.cpu(), indices.cpu())]

    def predict_file(self, path: str, topk: int = 5):
        result = self.predict(path, topk)
        return {"path": path, **result[0], "topk": result}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="single image file")
    source.add_argument("--dir", help="folder of images")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output", default=None, help="CSV path for batch mode")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    predictor = Predictor(args.checkpoint)
    print(f"checkpoint {predictor.checkpoint} | device {predictor.device}")

    if args.image:
        result = predictor.predict_file(args.image, args.topk)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=1))
            return 0
        print(f"\n{os.path.basename(args.image)}")
        for rank, item in enumerate(result["topk"], 1):
            bar = "#" * int(item["probability"] * 40)
            print(f"  {rank}. {item['label']:<32} {item['probability']:.3f} {bar}")
        return 0

    paths = sorted(
        p for ext in IMAGE_EXTENSIONS
        for p in glob.glob(os.path.join(args.dir, f"*{ext}"))
    )
    if not paths:
        raise SystemExit(f"no images found in {args.dir}")

    rows = []
    for path in paths:
        result = predictor.predict_file(path, 1)
        rows.append({
            "file": os.path.basename(path),
            "label": result["label"],
            "probability": result["probability"],
            "class_index": result["class_index"],
        })

    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(paths[0])), "predictions.csv")
    with open(output, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "label", "probability", "class_index"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} images -> {output}")
    for row in rows[:5]:
        print(f"  {row['file']:<40} {row['label']:<28} {row['probability']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
