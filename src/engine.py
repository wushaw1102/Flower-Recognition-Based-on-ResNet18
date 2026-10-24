"""Training and evaluation loops shared by every entry point.

``first.py`` and ``fourth.py`` in ``legacy/`` each carried their own copy of the
loop; this module is the single version, with the extras those scripts grew
later (AMP, ReduceLROnPlateau, early stopping) folded in behind flags.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn

from .metrics import topk_correct


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    top3: float
    top5: float
    lr: float
    seconds: float


@dataclass
class History:
    rows: List[EpochStats] = field(default_factory=list)

    def append(self, stats: EpochStats) -> None:
        self.rows.append(stats)

    @property
    def best_val_acc(self) -> float:
        return max((r.val_acc for r in self.rows), default=0.0)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              max_batches: Optional[int] = None) -> Dict[str, float]:
    """One pass over ``loader``.

    Passing ``optimizer`` switches to training mode; omitting it evaluates.
    ``max_batches`` exists for smoke tests (`--limit-batches`).
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    top3 = 0
    top5 = 0
    seen = 0

    for step, (inputs, targets) in enumerate(loader):
        if max_batches and step >= max_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(inputs)
                loss = criterion(logits, targets)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.inference_mode():
                logits = model(inputs)
                loss = criterion(logits, targets)

        batch = targets.size(0)
        total_loss += float(loss.item()) * batch
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        top3 += topk_correct(logits.float(), targets, 3)
        top5 += topk_correct(logits.float(), targets, 5)
        seen += batch

    seen = seen or 1
    return {
        "loss": total_loss / seen,
        "acc": correct / seen,
        "top3": top3 / seen,
        "top5": top5 / seen,
        "samples": seen,
    }


def fit(model, train_loader, valid_loader, criterion, optimizer, device,
        epochs: int, scheduler=None, scaler=None, patience: int = 0,
        checkpoint_path: Optional[str] = None,
        history: Optional[History] = None) -> History:
    """Run ``epochs`` epochs, keeping the best validation weights on disk.

    ``patience > 0`` enables early stopping on validation accuracy.
    """
    history = history or History()
    best_acc = history.best_val_acc
    best_state = None
    stale = 0
    start = time.time()

    for epoch in range(epochs):
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_stats = run_epoch(model, valid_loader, criterion, device)

        if scheduler is not None:
            # ReduceLROnPlateau wants the monitored value; StepLR wants nothing.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_stats["acc"])
            else:
                scheduler.step()

        stats = EpochStats(
            epoch=epoch,
            train_loss=train_stats["loss"], train_acc=train_stats["acc"],
            val_loss=val_stats["loss"], val_acc=val_stats["acc"],
            top3=val_stats["top3"], top5=val_stats["top5"],
            lr=optimizer.param_groups[0]["lr"],
            seconds=time.time() - start,
        )
        history.append(stats)

        print(f"epoch {epoch:>3} | train {stats.train_loss:.4f} / {stats.train_acc:.4f} "
              f"| val {stats.val_loss:.4f} / {stats.val_acc:.4f} "
              f"| top5 {stats.top5:.4f} | lr {stats.lr:.1e} | {stats.seconds:.0f}s",
              flush=True)

        if stats.val_acc > best_acc:
            best_acc = stats.val_acc
            stale = 0
            if checkpoint_path:
                torch.save({
                    "state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "epoch": epoch,
                }, checkpoint_path)
                print(f"  saved -> {checkpoint_path} (val_acc {best_acc:.4f})", flush=True)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if patience and stale >= patience:
                print(f"early stop: no improvement for {patience} epochs", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate(model, loader, criterion, device, max_batches: Optional[int] = None):
    """One inference pass; returns ``(stats, y_true, y_pred, logits)``.

    Everything the reports need comes out of this single pass — running a second
    loop for the metrics would double the cost for no reason.
    """
    model.eval()
    y_true, y_pred, all_logits = [], [], []
    total_loss = 0.0
    correct = top3 = top5 = seen = 0

    with torch.inference_mode():
        for step, (inputs, targets) in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)

            batch = targets.size(0)
            total_loss += float(loss.item()) * batch
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            top3 += topk_correct(logits.float(), targets, 3)
            top5 += topk_correct(logits.float(), targets, 5)
            seen += batch

            all_logits.append(logits.float().cpu())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(targets.cpu().tolist())

    seen = seen or 1
    stats = {"loss": total_loss / seen, "acc": correct / seen,
             "top3": top3 / seen, "top5": top5 / seen, "samples": seen}
    logits_tensor = torch.cat(all_logits) if all_logits else torch.empty(0)
    return stats, y_true, y_pred, logits_tensor
