"""Uncertainty and hypothesis-testing utilities for small classification test sets (NumPy + stdlib).

* Wilson score interval for a single accuracy (Wilson, 1927; Brown et al., 2001).
* Percentile bootstrap intervals for any metric, and paired bootstrap for differences between models.
* Exact McNemar test for paired classifiers (McNemar, 1947; Dietterich, 1998).
* Holm step-down correction for multiple comparisons (Holm, 1979).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import comb, sqrt
from statistics import NormalDist
from typing import Any

import numpy as np


def _z(confidence: float) -> float:
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    z = _z(confidence)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    low = 0.0 if successes == 0 else max(0.0, centre - half)  # exact bounds at the edges (no rounding residue)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return low, high


def bootstrap_ci(
    statistic: Callable[..., float],
    arrays: Sequence[np.ndarray],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap interval; ``arrays`` are resampled jointly (same indices) to keep pairs aligned."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = float(statistic(*arrays))
    draws = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        draws[b] = statistic(*(a[idx] for a in arrays))
    alpha = (1 - confidence) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    return {"estimate": point, "low": float(low), "high": float(high), "se": float(draws.std(ddof=1))}


def paired_bootstrap_difference(
    correct_a: Sequence[bool], correct_b: Sequence[bool], n_resamples: int = 10_000, confidence: float = 0.95, seed: int = 0
) -> dict[str, float]:
    """Bootstrap interval for accuracy(A) - accuracy(B) on the same test images."""
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    return bootstrap_ci(lambda x, y: float(x.mean() - y.mean()), [a, b], n_resamples, confidence, seed)


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, Any]:
    """Exact (binomial) McNemar test on the discordant pairs of two classifiers.

    ``b`` = images A got right and B got wrong; ``c`` = the reverse. Under H0 (equal error rates)
    b ~ Binomial(b + c, 0.5). Returns the two-sided p-value.
    """
    a = np.asarray(correct_a, dtype=bool)
    c_ = np.asarray(correct_b, dtype=bool)
    if a.shape != c_.shape:
        raise ValueError("Both classifiers must be evaluated on the same images")
    b = int(np.sum(a & ~c_))
    c = int(np.sum(~a & c_))
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
        p_value = min(1.0, 2 * tail)
    return {"a_only_correct": b, "b_only_correct": c, "discordant": n, "p_value": float(p_value)}


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1), in the original order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()
