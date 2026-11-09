"""Image-level data audit: integrity, geometry, formats, exact and near duplicates, cross-split leakage.

Only facts measured from the files are reported; nothing is inferred about image content.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .data import IMAGE_EXTENSIONS


def dhash(image: Image.Image, size: int = 8) -> int:
    """64-bit difference hash: robust to resizing and recompression, sensitive to content changes."""
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int(sum(1 << i for i, bit in enumerate(bits) if bit))


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(a ^ b).count("1")


def _describe(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {}
    return {"min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max()), "mean": float(arr.mean())}


def scan_images(root: str | Path, splits: Sequence[str] = ("train", "test")) -> pd.DataFrame:
    """One row per file with its split, class, geometry, format, size, hashes and integrity status."""
    rows = []
    for split in splits:
        split_dir = Path(root) / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for path in sorted(class_dir.iterdir()):
                if not path.is_file():
                    continue
                row: dict[str, Any] = {
                    "split": split,
                    "label": class_dir.name,
                    "path": str(path.relative_to(root)),
                    "extension": path.suffix.lower(),
                    "file_kb": round(path.stat().st_size / 1024, 1),
                    "md5": hashlib.md5(path.read_bytes()).hexdigest(),
                    "supported_extension": path.suffix.lower() in IMAGE_EXTENSIONS,
                }
                try:
                    with Image.open(path) as probe:
                        probe.verify()  # structural check
                    with Image.open(path) as image:
                        image.load()  # full decode catches truncated files
                        row.update(width=image.width, height=image.height, mode=image.mode, format=image.format)
                        row["aspect_ratio"] = round(image.width / image.height, 4)
                        row["dhash"] = f"{dhash(image):016x}"
                    row["corrupted"] = False
                except Exception as error:  # unreadable or truncated image
                    row.update(corrupted=True, error=type(error).__name__)
                rows.append(row)
    return pd.DataFrame(rows)


def find_near_duplicates(frame: pd.DataFrame, max_distance: int = 6) -> list[dict[str, Any]]:
    """Pairs of readable images whose difference hashes differ in at most ``max_distance`` bits."""
    readable = frame[~frame["corrupted"]].reset_index(drop=True)
    hashes = [int(h, 16) for h in readable["dhash"]]
    pairs = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            distance = hamming(hashes[i], hashes[j])
            if distance <= max_distance:
                a, b = readable.loc[i], readable.loc[j]
                pairs.append(
                    {
                        "a": a["path"],
                        "b": b["path"],
                        "hamming": distance,
                        "exact": a["md5"] == b["md5"],
                        "cross_split": a["split"] != b["split"],
                        "same_label": a["label"] == b["label"],
                    }
                )
    return pairs


def audit_dataset(root: str | Path, max_distance: int = 6) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run the full audit and return a JSON-serialisable summary plus the per-image table."""
    frame = scan_images(root)
    if frame.empty:
        raise ValueError(f"No images found under {root}")
    if "width" not in frame:  # every file corrupted: keep the columns the summary expects
        frame = frame.assign(width=np.nan, height=np.nan, aspect_ratio=np.nan, mode=None, format=None, dhash=None)
    readable = frame[~frame["corrupted"]]
    duplicates = find_near_duplicates(frame, max_distance)
    counts = {split: group["label"].value_counts().sort_index().to_dict() for split, group in frame.groupby("split")}
    summary: dict[str, Any] = {
        "n_images": int(len(frame)),
        "classes": sorted(frame["label"].unique().tolist()),
        "counts": counts,
        "n_corrupted": int(frame["corrupted"].sum()),
        "corrupted_files": frame.loc[frame["corrupted"], "path"].tolist(),
        "unsupported_files": frame.loc[~frame["supported_extension"], "path"].tolist(),
        "formats": dict(Counter(readable["format"].dropna())),
        "modes": dict(Counter(readable["mode"].dropna())),
        "width": _describe(readable["width"].dropna()),
        "height": _describe(readable["height"].dropna()),
        "aspect_ratio": _describe(readable["aspect_ratio"].dropna()),
        "file_kb": _describe(frame["file_kb"]),
        "near_duplicate_threshold_bits": max_distance,
        "exact_duplicate_pairs": sum(p["exact"] for p in duplicates),
        "near_duplicate_pairs": sum(not p["exact"] for p in duplicates),
        "cross_split_pairs": sum(p["cross_split"] for p in duplicates),
        "conflicting_label_pairs": sum(not p["same_label"] for p in duplicates),
        "duplicate_pairs": duplicates,
    }
    return summary, frame
