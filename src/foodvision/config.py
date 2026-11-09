"""Typed experiment configuration loaded from YAML, with a stable fingerprint for cache invalidation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    name: str = "pizza_steak_sushi_20_percent"
    root: str = "data"
    val_fraction: float = 0.2
    urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    drop_last: bool = False
    freeze_bn_stats: bool = False


@dataclass(frozen=True)
class ResearchConfig:
    seeds: tuple[int, ...] = (42, 43, 44)
    bn_ablation: bool = True


@dataclass(frozen=True)
class EvalConfig:
    bootstrap_resamples: int = 10_000
    calibration_bins: int = 10
    gallery_size: int = 12
    latency_runs: int = 30
    export_gallery: bool = True


@dataclass(frozen=True)
class OODConfig:
    """Not-food check: public non-food photos are scored to choose and test a rejection threshold."""

    enabled: bool = True
    sources: tuple[str, ...] = ("flowers102", "oxford_pets")
    per_source: int = 200
    target_tpr: float = 0.95


@dataclass(frozen=True)
class AnalysisConfig:
    """Robustness test, feature map and data-efficiency curve for the selected model."""

    fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
    perplexity: float = 30.0


@dataclass(frozen=True)
class PathsConfig:
    runs: str = "runs"
    artifacts: str = "artifacts"
    checkpoints: str = "checkpoints"


@dataclass(frozen=True)
class Config:
    seed: int = 42
    models: tuple[str, ...] = ("effnetb2", "effnetb3", "effnetb4")
    epochs: tuple[int, ...] = (15, 20, 30)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    ood: OODConfig = field(default_factory=OODConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self, *sections: str) -> str:
        """Short SHA-256 of the given top-level sections (all sections when none are given)."""
        payload = self.to_dict()
        if sections:
            payload = {key: payload[key] for key in sections}
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


_SECTIONS: dict[str, type[Any]] = {
    "data": DataConfig,
    "train": TrainConfig,
    "research": ResearchConfig,
    "evaluation": EvalConfig,
    "ood": OODConfig,
    "analysis": AnalysisConfig,
    "paths": PathsConfig,
}


def _build(cls: type[Any], values: dict[str, Any]) -> Any:
    """Instantiate a dataclass, rejecting unknown keys (typos) and converting lists to tuples."""
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})


def load_config(path: str | Path) -> Config:
    """Read a YAML file into a validated :class:`Config`."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    sections = {name: _build(cls, raw.pop(name, {}) or {}) for name, cls in _SECTIONS.items()}
    config: Config = _build(Config, {**raw, **sections})
    if not 0.0 < config.data.val_fraction < 1.0:
        raise ValueError("data.val_fraction must be between 0 and 1")
    if config.seed not in config.research.seeds:
        raise ValueError("seed must be one of research.seeds (the final model uses the main seed's run)")
    if config.train.batch_size < 2:
        raise ValueError("train.batch_size must be at least 2 (BatchNorm needs more than one sample)")
    if config.ood.enabled:
        from .ood import SOURCES

        if not config.ood.sources or set(config.ood.sources) - set(SOURCES):
            raise ValueError(f"ood.sources must be a non-empty subset of {sorted(SOURCES)}")
        if config.ood.per_source < 4:
            raise ValueError("ood.per_source must be at least 4 (half validation, half test)")
        if not 0.5 <= config.ood.target_tpr < 1.0:
            raise ValueError("ood.target_tpr must be at least 0.5 and below 1")
    if not config.analysis.fractions or not all(0.0 < f <= 1.0 for f in config.analysis.fractions):
        raise ValueError("analysis.fractions must be a non-empty list of values in (0, 1]")
    if config.analysis.perplexity < 2:
        raise ValueError("analysis.perplexity must be at least 2")
    return config
