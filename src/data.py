"""Dataset plumbing: transforms, ``ImageFolder`` wrappers and the demo index."""

import csv
import os
import random
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    RESIZE_SIZE,
    SEED,
    SAMPLE_IMAGES,
    TRAIN_DIR,
    VALID_DIR,
)


def train_transform(resize=RESIZE_SIZE, crop=INPUT_SIZE):
    """Augmentation stack used to train the published checkpoint.

    The order matters: rotate on the larger canvas, then crop, so the crop never
    exposes the zero-padded corners that rotation introduces.
    """
    return transforms.Compose([
        transforms.Resize([resize, resize]),
        transforms.RandomRotation(45),
        transforms.CenterCrop(crop),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(p=0.025),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size=INPUT_SIZE):
    """Deterministic resize used for validation, test and inference."""
    return transforms.Compose([
        transforms.Resize([size, size]),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_datasets(train_dir=TRAIN_DIR, valid_dir=VALID_DIR, size=INPUT_SIZE):
    """Return ``(train_dataset, valid_dataset)``.

    Both splits must expose the same ``classes`` list — ``ImageFolder`` derives
    it from sorted folder names, and the numeric folder names ('1', '10', '100',
    ...) sort lexicographically. That is fine as long as the two folders agree,
    and ``assert`` guards against a silently mismatched pair.
    """
    train_ds = datasets.ImageFolder(train_dir, train_transform(crop=size))
    valid_ds = datasets.ImageFolder(valid_dir, eval_transform(size))
    assert train_ds.classes == valid_ds.classes, "train/valid class folders differ"
    return train_ds, valid_ds


def build_loaders(train_ds, valid_ds, batch_size=128, workers=4):
    """DataLoaders with a seeded generator so shuffling is repeatable."""
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, generator=generator,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )
    return train_loader, valid_loader


def limit(dataset, count, seed=SEED):
    """Take a reproducible stratified-ish subset (used by ``--limit`` flags)."""
    if not count or count >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), count)
    return Subset(dataset, indices)


def class_distribution(dataset):
    """``{class_id: image count}`` for an ``ImageFolder`` (or its ``Subset``)."""
    base = getattr(dataset, "dataset", dataset)
    targets = getattr(base, "targets", None)
    if targets is None:
        return {}
    if hasattr(dataset, "indices"):
        targets = [targets[i] for i in dataset.indices]
    return {base.classes[k]: v for k, v in sorted(Counter(targets).items())}


def load_sample_index(path=None):
    """Read ``datasets/sample/index.csv`` -> list of dicts for the demo gallery."""
    path = path or os.path.join(SAMPLE_IMAGES, "..", "index.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            image_path = os.path.join(SAMPLE_IMAGES, row["file_name"])
            if os.path.exists(image_path):
                rows.append({**row, "path": image_path})
    return rows
