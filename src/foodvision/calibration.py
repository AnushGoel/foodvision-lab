"""Probability calibration: temperature scaling, expected calibration error, reliability bins, abstention.

Temperature scaling (Guo et al., 2017) divides the logits by one scalar T fitted on held-out validation data
by minimising the negative log-likelihood; it changes confidence but never the predicted class.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood (cross-entropy) of the true labels."""
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), np.asarray(labels, dtype=int)].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray, low: float = 0.05, high: float = 20.0) -> float:
    """Temperature minimising validation NLL, by golden-section search on log T (NLL is unimodal in T)."""
    a, b = np.log(low), np.log(high)
    ratio = (np.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    for _ in range(80):
        if nll(logits, labels, float(np.exp(c))) < nll(logits, labels, float(np.exp(d))):
            b, d = d, c
            c = b - ratio * (b - a)
        else:
            a, c = c, d
            d = a + ratio * (b - a)
    return float(np.exp((a + b) / 2))


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Equal-width confidence bins with their mean confidence, accuracy and count."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence > lo) & (confidence <= hi) if i else (confidence >= lo) & (confidence <= hi)
        count = int(mask.sum())
        bins.append(
            {
                "low": float(lo),
                "high": float(hi),
                "count": count,
                "confidence": float(confidence[mask].mean()) if count else float("nan"),
                "accuracy": float(correct[mask].mean()) if count else float("nan"),
            }
        )
    return bins


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum over bins of (bin share) x |accuracy - mean confidence|."""
    n = len(labels)
    return float(
        sum(b["count"] / n * abs(b["accuracy"] - b["confidence"]) for b in reliability_bins(probs, labels, n_bins) if b["count"])
    )


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between the probability vector and the one-hot label."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(probs.shape[1])[np.asarray(labels, dtype=int)]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def abstention_curve(probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray | None = None) -> list[dict[str, float]]:
    """Coverage and selective accuracy when predictions below a confidence threshold are withheld."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    thresholds = np.linspace(0.34, 0.99, 66) if thresholds is None else thresholds
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    rows = []
    for t in thresholds:
        kept = confidence >= t
        rows.append(
            {
                "threshold": float(t),
                "coverage": float(kept.mean()),
                "accuracy": float(correct[kept].mean()) if kept.any() else float("nan"),
            }
        )
    return rows


def calibration_summary(
    val_logits: np.ndarray, val_labels: np.ndarray, test_logits: np.ndarray, test_labels: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """Fit T on validation, then report test calibration before and after scaling."""
    temperature = fit_temperature(val_logits, val_labels)
    out: dict[str, Any] = {"temperature": temperature, "n_bins": n_bins}
    for name, t in (("uncalibrated", 1.0), ("temperature_scaled", temperature)):
        probs = softmax(test_logits, t)
        out[name] = {
            "ece": expected_calibration_error(probs, test_labels, n_bins),
            "nll": nll(test_logits, test_labels, t),
            "brier": brier_score(probs, test_labels),
            "bins": reliability_bins(probs, test_labels, n_bins),
        }
    return out
