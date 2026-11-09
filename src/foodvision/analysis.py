"""Torch-free helpers for the analysis stage: photo corruptions, a 2D feature map and data-efficiency subsets.

* Robustness: five everyday corruptions at three severities, applied to the model's view of each test photo (after
  resizing and cropping), in the spirit of ImageNet-C (Hendrycks & Dietterich, 2019).
* Feature map: penultimate-layer features, L2-normalised, reduced with PCA and projected to two dimensions with t-SNE
  (van der Maaten & Hinton, 2008). Neighbourhoods in a t-SNE map are meaningful; distances between clusters are not.
* Data efficiency: class-stratified subsets of the training split that are nested, so for one seed a larger fraction
  always contains the smaller ones and the curve reflects the amount of data rather than a new draw.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

CORRUPTIONS: dict[str, dict[str, Any]] = {
    "blur": {"label": "Blur", "levels": (1.0, 2.0, 3.5), "unit": "Gaussian radius, px"},
    "noise": {"label": "Sensor noise", "levels": (0.04, 0.08, 0.16), "unit": "noise SD on a 0-1 scale"},
    "dark": {"label": "Low light", "levels": (0.6, 0.4, 0.25), "unit": "brightness factor"},
    "jpeg": {"label": "JPEG compression", "levels": (30, 15, 7), "unit": "JPEG quality"},
    "lowres": {"label": "Low resolution", "levels": (2, 4, 8), "unit": "downscale factor"},
}
SEVERITIES = (1, 2, 3)


def conditions() -> list[tuple[str, int]]:
    """The clean baseline followed by every corruption at every severity."""
    return [("none", 0)] + [(kind, severity) for kind in CORRUPTIONS for severity in SEVERITIES]


def corrupt(image: Image.Image, kind: str, severity: int, rng: np.random.Generator) -> Image.Image:
    """Apply one corruption; severity 0 (or kind ``none``) returns an unchanged copy. The size never changes."""
    image = image.convert("RGB")
    if kind == "none" or severity == 0:
        return image.copy()
    if kind not in CORRUPTIONS or severity not in SEVERITIES:
        raise ValueError(f"Unknown corruption {kind!r} at severity {severity}")
    level = CORRUPTIONS[kind]["levels"][severity - 1]
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=level))
    if kind == "noise":
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        noisy = np.clip(pixels + rng.normal(0.0, level, pixels.shape), 0.0, 1.0)
        return Image.fromarray((noisy * 255.0 + 0.5).astype(np.uint8))
    if kind == "dark":
        return ImageEnhance.Brightness(image).enhance(level)
    if kind == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(level))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")
    width, height = image.size  # lowres: shrink, then enlarge back to the same size
    small = image.resize((max(1, width // int(level)), max(1, height // int(level))), Image.Resampling.BILINEAR)
    return small.resize((width, height), Image.Resampling.BILINEAR)


def nested_subset(labels: Sequence[int], fraction: float, seed: int) -> np.ndarray:
    """Sorted indices of a class-stratified subset; for one seed, smaller fractions are contained in larger ones."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    chosen = []
    for label in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == label))
        chosen.append(members[: max(1, int(round(len(members) * fraction)))])
    return np.sort(np.concatenate(chosen))


def project_2d(features: np.ndarray, seed: int = 0, perplexity: float = 30.0) -> np.ndarray:
    """2D coordinates for a feature map: L2 normalisation, PCA to at most 50 dimensions, then t-SNE."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    x = np.asarray(features, dtype=np.float64)
    if len(x) < 5:
        raise ValueError("A feature map needs at least five images")
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    reduced = PCA(n_components=min(50, len(x) - 1, x.shape[1]), random_state=seed).fit_transform(x)
    tsne = TSNE(n_components=2, perplexity=min(perplexity, (len(x) - 1) / 3), init="pca", learning_rate="auto",
                random_state=seed)
    return np.asarray(tsne.fit_transform(reduced))


def efficiency_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of validation accuracy per training fraction."""
    return (
        frame.groupby("fraction")
        .agg(n_train=("n_train", "mean"), val_acc_mean=("val_acc", "mean"), val_acc_sd=("val_acc", "std"),
             runs=("val_acc", "size"))
        .reset_index()
        .sort_values("fraction")
    )
