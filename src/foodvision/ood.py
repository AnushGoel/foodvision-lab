"""Telling pizza, steak and sushi apart from everything else (out-of-distribution detection).

A closed-set classifier must answer with one of its classes, so a photo of a dog still gets a label. Three standard
post-hoc scores turn the logits into one number that is higher for images that resemble the training data:

* maximum softmax probability, MSP (Hendrycks & Gimpel, 2017), computed with the deployed temperature;
* maximum logit (Hendrycks et al., 2022);
* negative free energy, the log-sum-exp of the logits (Liu et al., 2020).

The score and its threshold are chosen on validation data: the in-distribution validation images plus a validation
sample of non-food photos. The test images of both kinds are then used once to report the result. NumPy only, so
the app can score uploads without extra dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import softmax

SCORES = ("msp", "max_logit", "energy")
SCORE_LABELS = {
    "msp": "Maximum softmax probability",
    "max_logit": "Maximum logit",
    "energy": "Negative energy (log-sum-exp)",
}
# Public non-food photo collections, downloaded through torchvision. Only scores are kept, never the images.
SOURCES: dict[str, dict[str, str]] = {
    "flowers102": {"label": "Flowers-102", "content": "flowers", "citation": "Nilsback & Zisserman, 2008",
                   "dataset": "Flowers102", "pattern": "flowers-102/jpg/*.jpg"},
    "oxford_pets": {"label": "Oxford-IIIT Pet", "content": "cats and dogs", "citation": "Parkhi et al., 2012",
                    "dataset": "OxfordIIITPet", "pattern": "oxford-iiit-pet/images/*.jpg"},
}


def scores(logits: np.ndarray, temperature: float = 1.0) -> dict[str, np.ndarray]:
    """All three scores for a batch of logits (higher = more like the training data)."""
    z = np.asarray(logits, dtype=float)
    top = z.max(axis=1, keepdims=True)
    log_sum_exp = (top + np.log(np.exp(z - top).sum(axis=1, keepdims=True)))[:, 0]
    return {"msp": softmax(z, temperature).max(axis=1), "max_logit": z.max(axis=1), "energy": log_sum_exp}


def single_score(logits: Sequence[float], kind: str, temperature: float = 1.0) -> float:
    """One score for one image's logits, as used by the app."""
    if kind not in SCORES:
        raise ValueError(f"Unknown score {kind!r}; expected one of {SCORES}")
    return float(scores(np.asarray(logits, dtype=float)[None, :], temperature)[kind][0])


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Probability that a random in-distribution image outscores a random non-food image (ties count half)."""
    x = np.asarray(id_scores, dtype=float)
    y = np.asarray(ood_scores, dtype=float)
    if not len(x) or not len(y):
        return float("nan")
    diff = x[:, None] - y[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def threshold_for_tpr(id_scores: np.ndarray, target_tpr: float) -> float:
    """Highest threshold that still accepts at least ``target_tpr`` of the in-distribution images (accept: score >= t)."""
    s = np.sort(np.asarray(id_scores, dtype=float))
    if not len(s):
        raise ValueError("Need at least one in-distribution score")
    rejected = int(np.floor((1.0 - target_tpr) * len(s) + 1e-9))
    return float(s[min(rejected, len(s) - 1)])


def acceptance(scores_: np.ndarray, threshold: float) -> float:
    """Share of images whose score reaches the threshold."""
    s = np.asarray(scores_, dtype=float)
    return float((s >= threshold).mean()) if len(s) else float("nan")


def fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """Share of non-food images still accepted when the threshold keeps ``target_tpr`` of the food images."""
    return acceptance(ood_scores, threshold_for_tpr(id_scores, target_tpr))


def auroc_interval(
    id_scores: np.ndarray, ood_scores: np.ndarray, n_resamples: int = 2000, level: float = 0.95, seed: int = 0
) -> dict[str, float]:
    """AUROC with a percentile bootstrap interval that resamples the two groups separately."""
    x = np.asarray(id_scores, dtype=float)
    y = np.asarray(ood_scores, dtype=float)
    rng = np.random.default_rng(seed)
    draws = [auroc(rng.choice(x, len(x)), rng.choice(y, len(y))) for _ in range(n_resamples)]
    alpha = (1.0 - level) / 2
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return {"estimate": auroc(x, y), "low": float(low), "high": float(high), "level": level, "n_resamples": n_resamples}


def sample_split(
    files: Mapping[str, Sequence[Path]], per_source: int, seed: int
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Random, reproducible sample of each source, halved into validation and test parts (no overlap)."""
    rng = np.random.default_rng(seed)
    val: list[tuple[Path, str]] = []
    test: list[tuple[Path, str]] = []
    for source in sorted(files):
        pool = sorted(files[source])
        n = min(per_source, len(pool))
        chosen = [pool[i] for i in rng.choice(len(pool), n, replace=False)]
        half = n // 2
        val.extend((path, source) for path in chosen[:half])
        test.extend((path, source) for path in chosen[half:])
    return val, test


def evaluate(
    val_id_logits: np.ndarray,
    val_ood_logits: np.ndarray,
    test_id_logits: np.ndarray,
    test_ood_logits: np.ndarray,
    test_ood_sources: Sequence[str],
    temperature: float = 1.0,
    target_tpr: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Choose the score and threshold on validation data, then report them once on the test data."""
    val_id, val_ood = scores(val_id_logits, temperature), scores(val_ood_logits, temperature)
    test_id, test_ood = scores(test_id_logits, temperature), scores(test_ood_logits, temperature)
    table: list[dict[str, Any]] = [
        {
            "score": name,
            "label": SCORE_LABELS[name],
            "val_auroc": auroc(val_id[name], val_ood[name]),
            "val_fpr_at_tpr": fpr_at_tpr(val_id[name], val_ood[name], target_tpr),
            "test_auroc": auroc(test_id[name], test_ood[name]),
            "test_fpr_at_tpr": fpr_at_tpr(test_id[name], test_ood[name], target_tpr),
        }
        for name in SCORES
    ]
    best = max(table, key=lambda r: (r["val_auroc"], -r["val_fpr_at_tpr"], -SCORES.index(r["score"])))
    name = best["score"]
    threshold = threshold_for_tpr(val_id[name], target_tpr)
    sources = np.asarray(test_ood_sources)
    per_source: list[dict[str, Any]] = [
        {
            "source": source,
            "label": f"{SOURCES[source]['label']} ({SOURCES[source]['content']})" if source in SOURCES else source,
            "n": int((sources == source).sum()),
            "rejected": 1.0 - acceptance(test_ood[name][sources == source], threshold),
        }
        for source in sorted(set(sources.tolist()))
    ]
    return {
        "status": "ok",
        "score": name,
        "score_label": SCORE_LABELS[name],
        "temperature": float(temperature),
        "target_tpr": float(target_tpr),
        "threshold": threshold,
        "selection_rule": "highest validation AUROC; ties: lower validation FPR at the target TPR, then the simpler score",
        "scores": table,
        "validation": {
            "n_food": len(val_id[name]),
            "n_non_food": len(val_ood[name]),
            "accepted_food": acceptance(val_id[name], threshold),
            "rejected_non_food": 1.0 - acceptance(val_ood[name], threshold),
        },
        "test": {
            "n_food": len(test_id[name]),
            "n_non_food": len(test_ood[name]),
            "auroc": auroc_interval(test_id[name], test_ood[name], n_resamples, seed=seed),
            "fpr_at_tpr": fpr_at_tpr(test_id[name], test_ood[name], target_tpr),
            "accepted_food": acceptance(test_id[name], threshold),
            "rejected_non_food": 1.0 - acceptance(test_ood[name], threshold),
        },
        "per_source": per_source,
        "protocol_note": "Score and threshold chosen on validation images (food) and a validation sample of non-food "
        "photos; the test images of both kinds were used once. Only obvious non-food (flowers, pets) was tested, so "
        "other foods are not covered by these numbers.",
    }


def score_table(
    logits: Mapping[str, np.ndarray], sources: Mapping[str, Sequence[str]], temperature: float = 1.0
) -> pd.DataFrame:
    """Per-image scores for plotting. Keys are ``<split>_<group>``, e.g. ``test_food`` or ``val_non_food``."""
    frames = []
    for key, z in logits.items():
        split, group = key.split("_", 1)
        values = scores(z, temperature)
        frames.append(
            pd.DataFrame(
                {"split": split, "group": group.replace("_", "-"), "source": list(sources[key]),
                 **{name: values[name] for name in SCORES}}
            )
        )
    return pd.concat(frames, ignore_index=True)
