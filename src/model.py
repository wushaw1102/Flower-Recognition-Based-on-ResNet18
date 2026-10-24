"""ResNet-18 with a 102-way head, plus checkpoint loading that accepts every
format this project has produced over time."""

import os
from typing import Optional, Tuple

import torch
from torch import nn
from torchvision import models

from .config import INPUT_SIZE, NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES,
                pretrained: bool = False,
                freeze_backbone: bool = False) -> nn.Module:
    """Return a ResNet-18 whose classifier outputs ``num_classes`` logits.

    ``pretrained=True`` downloads ImageNet weights (~45 MB) — needed for
    training, unnecessary for evaluation. If the download fails the model still
    builds, just from random initialisation, and the caller is told.
    """
    weights = None
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:                      # offline / blocked mirror
        print(f"[model] could not load ImageNet weights ({exc}); using random init")
        model = models.resnet18(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features          # 512 for resnet18
    model.fc = nn.Linear(in_features, num_classes)   # always trainable

    return model


def unfreeze(model: nn.Module) -> None:
    """Make every parameter trainable (stage 2 of the two-stage schedule)."""
    for param in model.parameters():
        param.requires_grad = True


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def _clean_state_dict(payload) -> Tuple[dict, Optional[float]]:
    """Unwrap the checkpoint formats seen in this repo.

    * ``{'state_dict': ..., 'best_acc': ..., 'optimizer': ...}`` — what
      ``first.py``/``fourth.py`` wrote as ``best.pt``;
    * a bare ``state_dict`` — what ``torch.save(model.state_dict())`` writes;
    * either of the above wrapped in ``DataParallel`` (``module.`` prefix).
    """
    best_acc = None
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                best_acc = payload.get("best_acc")
                if isinstance(best_acc, torch.Tensor):
                    best_acc = float(best_acc.item())
                payload = payload[key]
                break

    state_dict = {}
    for key, value in payload.items():
        state_dict[key[len("module."):] if key.startswith("module.") else key] = value
    return state_dict, best_acc


def load_checkpoint(path: str,
                    num_classes: int = NUM_CLASSES,
                    device: Optional[torch.device] = None) -> Tuple[nn.Module, Optional[float]]:
    """Load a trained checkpoint and return ``(model_in_eval_mode, best_acc)``."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes, pretrained=False)

    payload = torch.load(path, map_location="cpu")
    state_dict, best_acc = _clean_state_dict(payload)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[model] ignored {len(unexpected)} unexpected keys "
              f"(e.g. {unexpected[0]}) — expected for optimizer state")
    if missing:
        print(f"[model] {len(missing)} missing keys (e.g. {missing[0]})")

    model.to(device)
    model.eval()
    return model, best_acc


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
