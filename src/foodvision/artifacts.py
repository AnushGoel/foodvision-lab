"""The versioned artifact contract between the GPU pipeline (writer) and the Streamlit app (reader).

The app only reads these small JSON/CSV/image files, so it never needs to train or touch the dataset.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1


class Files:
    MANIFEST = "manifest.json"
    AUDIT = "data/audit.json"
    IMAGES = "data/images.csv"
    ARCHITECTURE = "models/architecture.json"
    BENCHMARK = "models/benchmark.json"
    ASSIGNMENT_SUMMARY = "assignment/summary.csv"
    ASSIGNMENT_HISTORY = "assignment/histories.json"
    VALIDATION = "research/validation.csv"
    RESEARCH_HISTORY = "research/histories.json"
    SELECTION = "research/selection.json"
    TEST_REPORT = "research/test_report.json"
    PREDICTIONS = "research/predictions_test.csv"
    CALIBRATION = "research/calibration.json"
    BN_ABLATION = "research/bn_ablation.csv"
    OOD = "research/ood.json"
    OOD_SCORES = "research/ood_scores.csv"
    GALLERY = "gallery/index.json"
    ROBUSTNESS = "analysis/robustness.csv"
    FEATURE_MAP = "analysis/feature_map.csv"
    DATA_EFFICIENCY = "analysis/data_efficiency.csv"
    ANALYSIS = "analysis/summary.json"
    SUMMARY_FIGURE = "figures/summary.png"
    MODEL = "model/foodvision.pt"
    MODEL_META = "model/model_meta.json"
    MODEL_CARD = "reports/MODEL_CARD.md"
    EXPERIMENTS_REPORT = "reports/EXPERIMENTS.md"
    AUDIT_REPORT = "reports/DATA_AUDIT.md"


def default_root() -> Path:
    """``$FOODVISION_ARTIFACTS`` if set, otherwise ``<repo>/artifacts``."""
    env = os.environ.get("FOODVISION_ARTIFACTS")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "artifacts"


def _clean(obj: Any) -> Any:
    """Recursively convert to exact built-in types: dict, list, str, int, float, bool and None.

    Subclasses of built-ins become the exact built-in type. This matters beyond JSON: ``torch.__version__`` is a
    ``TorchVersion`` (a ``str`` subclass), and ``torch.load(weights_only=True)`` refuses to unpickle any class it does
    not know, so a checkpoint whose metadata held one could not be reloaded. NaN and infinity become None.
    """
    if obj is None or type(obj) is bool:
        return obj
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    return str(obj)  # str subclasses, Path, enums and any other object become plain text


def to_builtin(obj: Any) -> Any:
    """Plain Python containers and numbers only (also what ``torch.load(weights_only=True)`` accepts)."""
    return _clean(obj)


class ArtifactStore:
    """Read/write helper rooted at one artifacts directory; writes are atomic."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()

    def path(self, name: str) -> Path:
        return self.root / name

    def exists(self, name: str) -> bool:
        return self.path(name).is_file()

    def missing(self, *names: str) -> list[str]:
        return [n for n in names if not self.exists(n)]

    def _atomic_write(self, name: str, text: str) -> Path:
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)
        return target

    def write_json(self, name: str, obj: Any) -> Path:
        return self._atomic_write(name, json.dumps(_clean(obj), indent=2, allow_nan=False))

    def read_json(self, name: str) -> Any:
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def write_csv(self, name: str, frame: pd.DataFrame) -> Path:
        return self._atomic_write(name, frame.to_csv(index=False))

    def read_csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.path(name))

    def write_text(self, name: str, text: str) -> Path:
        return self._atomic_write(name, text)

    def manifest(self) -> dict[str, Any] | None:
        """The manifest, or None when the pipeline has not run or used an incompatible schema."""
        if not self.exists(Files.MANIFEST):
            return None
        manifest: dict[str, Any] = self.read_json(Files.MANIFEST)
        return manifest if manifest.get("schema_version") == SCHEMA_VERSION else None
