# Dataset

## What the model was trained on

The 102-category flower dataset that ships with the Udacity *PyTorch* course
(a subset of Oxford Flowers 102). Images are ~500 px on the short side, and the
splits are heavily class-imbalanced.

| Split | Classes | Images | Per-class range |
| :-- | --: | --: | --: |
| `flowers/train` | 102 | 6,552 | 20 – 100 |
| `flowers/valid` | 102 | 818 | 1 – 28 |

```
datasets/
├── cat_to_name.json      # "1" -> "pink primrose", 102 entries
├── classes.json          # ImageFolder's class order ('1', '10', '100', ...)
├── flowers/              # git-ignored, ~300 MB
│   ├── train/<id>/*.jpg
│   └── valid/<id>/*.jpg
└── sample/               # committed: 200 images, class-balanced
    ├── images/class<id>_<original>.jpg
    └── index.csv
```

`sample/` exists so the demo and the evaluation pipeline work straight after a
clone without downloading 300 MB. Rebuild or resize it with:

```bash
python scripts/make_sample.py --per-class 2      # 200 images, one per class x2
python scripts/make_sample.py --per-class 8      # every validation image
```

## Why `classes.json` matters

`torchvision.datasets.ImageFolder` sorts folder names **as strings**, so the
class index order is `1, 10, 100, 101, 102, 11, 12, ...` — not `1, 2, 3, ...`.
Mapping a predicted index with `str(index + 1)` therefore returns the wrong
flower for 101 of the 102 classes, and the bug is invisible during training
(the labels are consistent) and only shows up in a demo. `classes.json` records
the real order; load it through `src.config.load_class_order()`.

## Getting the full dataset

The Udacity course bundle (`flower_data.tar.gz`, ~330 MB) is the usual source.
Any layout with `train/<class>/*.jpg` and `valid/<class>/*.jpg` works — the code
reads folder names, not filenames. Once unpacked:

```bash
python -c "
import os, torchvision
ds = torchvision.datasets.ImageFolder('datasets/flowers/train')
print(len(ds), 'images,', len(ds.classes), 'classes')
"
```

## Licensing

Course images, not redistributed here. The 200-image `sample/` is included
solely to keep the demo runnable; treat it under the same terms as the original
dataset (research and educational use).
