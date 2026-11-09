"""Classification metrics computed from labels and predictions (NumPy only)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    """Counts with rows = true class and columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return cm


def normalize_confusion(cm: np.ndarray) -> np.ndarray:
    """Row-normalised confusion matrix (each row sums to 1, i.e. per-class recall on the diagonal)."""
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.zeros(cm.shape, dtype=float), where=totals > 0)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros(num.shape, dtype=float), where=den > 0)


def classification_report(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> dict[str, Any]:
    """Accuracy, per-class precision/recall/F1/support, and macro- and weighted-averaged F1."""
    k = len(class_names)
    cm = confusion_matrix(y_true, y_pred, k)
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    support = cm.sum(axis=1).astype(float)
    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    total = support.sum()
    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / total) if total else 0.0,
        "per_class": [
            {
                "class": name,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        ],
        "confusion_matrix": cm.tolist(),
    }


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> float:
    """Macro-averaged F1 (used as a bootstrap statistic)."""
    names = [str(i) for i in range(n_classes)]
    return float(classification_report(y_true, y_pred, names)["macro_f1"])
