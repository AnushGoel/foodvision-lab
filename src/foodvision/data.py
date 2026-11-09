"""Dataset download, indexing, stratified splitting and in-memory tensor caching.

Torch is imported lazily so the indexing and splitting utilities stay usable (and testable) without it.
"""

from __future__ import annotations

import logging
import random
import tarfile
import time
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

LOG = logging.getLogger(__name__)
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


@dataclass(frozen=True)
class Sample:
    """One labelled image on disk."""

    path: Path
    label: int


def download_dataset(root: str | Path, name: str, urls: Sequence[str], attempts: int = 5, timeout: int = 120) -> Path:
    """Download and extract the dataset once, trying every mirror with exponential back-off.

    A zip placed manually at ``<root>/<name>.zip`` is used when present, so the pipeline also works offline.
    """
    target = Path(root) / name
    if (target / "train").is_dir() and (target / "test").is_dir():
        return target
    target.mkdir(parents=True, exist_ok=True)
    archive = Path(root) / f"{name}.zip"
    if archive.is_file() and zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        return target
    for attempt in range(1, attempts + 1):
        for url in urls:
            try:
                LOG.info("Downloading %s (attempt %d)", url, attempt)
                request = urllib.request.Request(url, headers={"User-Agent": "foodvision-lab"})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    archive.write_bytes(response.read())
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(target)
                archive.unlink()
                return target
            except Exception as error:  # network hiccup or truncated zip: try the next mirror
                LOG.warning("Download failed: %s", error)
        time.sleep(min(60, 5 * 2**attempt))
    raise RuntimeError(f"Could not download the dataset after {attempts} attempts; place {archive} manually.")


# Download problems that should skip the not-food check instead of stopping the pipeline. Programming errors are not
# listed, so they still fail loudly (the two-minute smoke run in the Colab notebook catches them early).
DOWNLOAD_ERRORS = (OSError, RuntimeError, EOFError, tarfile.TarError, zipfile.BadZipFile)


def download_ood_source(root: str | Path, source: str, attempts: int = 3) -> list[Path]:
    """Download one public non-food photo collection via torchvision (once) and list its photos.

    Returns an empty list, with a warning, when the download keeps failing.
    """
    import torchvision.datasets as tv_datasets

    from .ood import SOURCES

    spec = SOURCES[source]
    folder = Path(root) / source
    for attempt in range(1, attempts + 1):
        try:
            getattr(tv_datasets, spec["dataset"])(str(folder), split="test", download=True)
            break
        except ImportError as error:  # e.g. scipy, needed only to read Flowers-102 labels: the photos are there
            LOG.warning("%s downloaded but its labels could not be read (%s); using the photos only", spec["label"], error)
            break
        except DOWNLOAD_ERRORS as error:
            LOG.warning("Download of %s failed (attempt %d of %d): %s", spec["label"], attempt, attempts, error)
            if attempt == attempts:
                return []
            time.sleep(10 * attempt)
    files = sorted(folder.glob(spec["pattern"]))
    if not files:  # folder layout changed in a newer torchvision: fall back to every photo outside annotations
        files = sorted(p for p in folder.rglob("*.jpg") if "annotations" not in p.parts and "trimaps" not in p.parts)
    LOG.info("%s: %d photos available", spec["label"], len(files))
    return files


def class_names(split_dir: str | Path) -> list[str]:
    """Class names are the sorted sub-folder names (the ImageFolder convention)."""
    return sorted(p.name for p in Path(split_dir).iterdir() if p.is_dir())


def index_split(split_dir: str | Path, classes: Sequence[str]) -> list[Sample]:
    """All images of one split, ordered by class and file name for reproducibility."""
    samples = []
    for label, name in enumerate(classes):
        files = sorted(p for p in (Path(split_dir) / name).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        samples.extend(Sample(p, label) for p in files)
    return samples


def stratified_split(samples: Sequence[Sample], val_fraction: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Split each class separately so the validation set keeps the class proportions."""
    rng = np.random.default_rng(seed)
    train: list[Sample] = []
    val: list[Sample] = []
    for label in sorted({s.label for s in samples}):
        group = [s for s in samples if s.label == label]
        order = rng.permutation(len(group))
        n_val = max(1, round(len(group) * val_fraction))
        val.extend(group[i] for i in order[:n_val])
        train.extend(group[i] for i in order[n_val:])
    return sorted(train, key=lambda s: str(s.path)), sorted(val, key=lambda s: str(s.path))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tensors(samples: Sequence[Sample], transform: Callable[[Any], Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode and transform every image once.

    The preprocessing is deterministic (no augmentation), so caching the tensors gives identical results to
    re-decoding each epoch while removing the CPU bottleneck of JPEG decoding on small cloud machines.
    """
    import torch
    from PIL import Image

    images = []
    for sample in samples:
        with Image.open(sample.path) as image:
            images.append(transform(image.convert("RGB")))
    return torch.stack(images), torch.tensor([s.label for s in samples], dtype=torch.long)


def make_loader(
    x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, seed: int = 0, drop_last: bool = False
) -> DataLoader:
    """DataLoader over cached tensors with a seeded shuffle order."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )
