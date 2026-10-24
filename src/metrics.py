"""Classification metrics without a scikit-learn dependency.

Everything takes plain integer lists so the same functions serve the training
loop (which has tensors) and the CLI evaluators (which have numpy arrays).
"""

from collections import defaultdict
from typing import Dict, List, Sequence


def topk_correct(logits, targets, k: int = 5) -> int:
    """Number of samples whose ground truth is inside the top-k predictions."""
    k = min(k, logits.shape[1])
    _, topk = logits.topk(k, dim=1)
    return int((topk == targets.view(-1, 1)).any(dim=1).sum().item())


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int],
                     num_classes: int) -> List[List[int]]:
    """``cm[true][pred]`` counts, row-major."""
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def per_class_accuracy(cm: List[List[int]]) -> Dict[int, float]:
    """Recall per class; classes with no samples are omitted."""
    out = {}
    for idx, row in enumerate(cm):
        total = sum(row)
        if total:
            out[idx] = row[idx] / total
    return out


def classification_report(y_true: Sequence[int], y_pred: Sequence[int],
                          num_classes: int) -> Dict[str, float]:
    """Macro and weighted precision / recall / F1 plus plain accuracy."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        t, p = int(t), int(p)
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    supports = defaultdict(int)
    for t in y_true:
        supports[int(t)] += 1

    precisions, recalls, f1s, weights = {}, {}, {}, {}
    for cls in range(num_classes):
        p_denom = tp[cls] + fp[cls]
        r_denom = tp[cls] + fn[cls]
        precision = tp[cls] / p_denom if p_denom else 0.0
        recall = tp[cls] / r_denom if r_denom else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        precisions[cls] = precision
        recalls[cls] = recall
        f1s[cls] = f1
        weights[cls] = supports[cls]

    total = sum(weights.values()) or 1

    def macro(values):
        present = [v for k, v in values.items() if weights[k] > 0]
        return sum(present) / len(present) if present else 0.0

    def weighted(values):
        return sum(v * weights[k] for k, v in values.items()) / total

    return {
        "accuracy": sum(1 for t, p in zip(y_true, y_pred) if int(t) == int(p)) / (len(y_true) or 1),
        "macro_precision": macro(precisions),
        "macro_recall": macro(recalls),
        "macro_f1": macro(f1s),
        "weighted_precision": weighted(precisions),
        "weighted_recall": weighted(recalls),
        "weighted_f1": weighted(f1s),
        "num_samples": len(y_true),
        "num_classes_present": sum(1 for v in weights.values() if v),
    }
