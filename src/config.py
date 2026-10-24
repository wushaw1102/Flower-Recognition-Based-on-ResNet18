"""Single source of truth for paths, class names and training defaults.

Every other module imports from here, so changing the image size or the dataset
location is a one-line edit rather than a hunt through four scripts.
"""

import json
import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.path.join(REPO_ROOT, "datasets")
FLOWER_ROOT = os.path.join(DATA_ROOT, "flowers")
TRAIN_DIR = os.path.join(FLOWER_ROOT, "train")
VALID_DIR = os.path.join(FLOWER_ROOT, "valid")
CAT_TO_NAME = os.path.join(DATA_ROOT, "cat_to_name.json")

SAMPLE_ROOT = os.path.join(DATA_ROOT, "sample")
SAMPLE_IMAGES = os.path.join(SAMPLE_ROOT, "images")
SAMPLE_INDEX = os.path.join(SAMPLE_ROOT, "index.csv")

WEIGHTS_DIR = os.path.join(REPO_ROOT, "weights")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")
DOCS_IMAGES = os.path.join(REPO_ROOT, "docs", "images")
LEGACY_DIR = os.path.join(REPO_ROOT, "legacy")

#: Checkpoints are tried in this order when a command runs without --checkpoint.
DEFAULT_CHECKPOINTS = [
    os.path.join(WEIGHTS_DIR, "flower_resnet18.pth"),
    os.path.join(OUTPUT_DIR, "resnet18-64", "best.pt"),
    os.path.join(WEIGHTS_DIR, "best.pt"),
]

# --------------------------------------------------------------------------- #
# Model / preprocessing constants
# --------------------------------------------------------------------------- #
NUM_CLASSES = 102
INPUT_SIZE = 64
RESIZE_SIZE = 96                     # train resizes to 96 then crops to 64

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#: The run that produced the published checkpoint (see README, "Results").
TWO_STAGE = {
    "stage1": {"epochs": 5, "lr": 1e-2, "frozen": True},
    "stage2": {"epochs": 5, "lr": 1e-3, "frozen": False},
}

BATCH_SIZE = 128
#: Windows spawns a fresh interpreter per worker (~5 s each, importing torch),
#: which dwarfs the cost of decoding 64x64 JPEGs. Forked platforms keep 4.
NUM_WORKERS = 0 if os.name == "nt" else 4
SEED = 42


def load_class_names():
    """Return ``{class_id: human name}`` from ``cat_to_name.json``.

    The JSON keys are the numeric folder names ('1'..'102').
    """
    with open(CAT_TO_NAME, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {str(k): v for k, v in raw.items()}


CLASS_ORDER = os.path.join(DATA_ROOT, "classes.json")


def load_class_order():
    """Folder names in the order ``ImageFolder`` assigns class indices.

    This is *not* the numeric order: ``ImageFolder`` sorts folder names as
    strings, so '1', '10', '100', '101', '102', '11', ... That difference is the
    reason a model can look correct in training and produce nonsense labels in a
    demo, so the order is recorded in ``datasets/classes.json`` the first time we
    see the dataset and reused everywhere afterwards.
    """
    if os.path.exists(CLASS_ORDER):
        with open(CLASS_ORDER, encoding="utf-8") as fh:
            return json.load(fh)

    from torchvision import datasets

    if os.path.isdir(TRAIN_DIR):
        return datasets.ImageFolder(TRAIN_DIR).classes
    return sorted(str(i) for i in range(1, NUM_CLASSES + 1))


def save_class_order(classes):
    """Persist the observed class order so inference and reports agree."""
    if os.path.exists(CLASS_ORDER):
        with open(CLASS_ORDER, encoding="utf-8") as fh:
            if json.load(fh) == list(classes):
                return
    os.makedirs(os.path.dirname(CLASS_ORDER), exist_ok=True)
    with open(CLASS_ORDER, "w", encoding="utf-8") as fh:
        json.dump(list(classes), fh, ensure_ascii=False, indent=1)


def default_checkpoint(explicit=None):
    """Resolve a checkpoint path, falling back through ``DEFAULT_CHECKPOINTS``."""
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"checkpoint not found: {explicit}")
        return explicit
    for candidate in DEFAULT_CHECKPOINTS:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "no checkpoint found. Download the released weights into weights/ "
        "(see README step 4) or train your own with `python -m src.train`."
    )
