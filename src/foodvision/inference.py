"""Validated single-image inference: calibrated probabilities, a not-food check, abstention and Grad-CAM.

``InvalidImageError`` and ``load_image`` are re-exported, so callers import validation and inference from one place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .explain import GradCAM, overlay
from .imageio import InvalidImageError, load_image  # noqa: F401
from .models import get_transforms, load_checkpoint
from .ood import SCORE_LABELS, single_score


@dataclass(frozen=True)
class Prediction:
    classes: list[str]
    probabilities: list[float]
    label: str
    confidence: float
    abstained: bool
    latency_ms: float
    ood_score: float | None = None
    in_distribution: bool | None = None  # None: no not-food check (old checkpoint, or switched off)


class Predictor:
    """Loads one delta checkpoint and serves predictions on CPU or GPU."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu", pretrained: bool = True) -> None:
        self.device = torch.device(device)
        self.model, payload = load_checkpoint(checkpoint, pretrained=pretrained, device=self.device)
        self.arch: str = payload["arch"]
        self.classes: list[str] = payload["classes"]
        self.meta: dict[str, Any] = payload["meta"]
        self.temperature = float(self.meta.get("temperature", 1.0))
        self.ood: dict[str, Any] | None = self.meta.get("ood")  # score name and threshold chosen on validation data
        self.transforms = get_transforms(self.arch)
        self._cam = GradCAM(self.model)

    def model_view(self, image: Image.Image) -> Image.Image:
        """The resized, centre-cropped view the network actually receives (before normalisation)."""
        resized = TF.resize(image, self.transforms.resize_size, interpolation=self.transforms.interpolation)
        return TF.center_crop(resized, self.transforms.crop_size)

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        return self.transforms(image).unsqueeze(0).to(self.device)

    @property
    def ood_label(self) -> str | None:
        return SCORE_LABELS.get(self.ood["score"]) if self.ood else None

    def predict(self, image: Image.Image, abstain_below: float = 0.0, check_ood: bool = True) -> Prediction:
        """Temperature-scaled class probabilities, the not-food verdict, and abstention below a confidence."""
        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(self._tensor(image))[0].float().cpu()
        probs = torch.softmax(logits / self.temperature, dim=0).numpy()
        ood_score: float | None = None
        in_distribution: bool | None = None
        if self.ood and check_ood:
            ood_score = single_score(logits.numpy(), self.ood["score"], self.temperature)
            in_distribution = ood_score >= float(self.ood["threshold"])
        latency = (time.perf_counter() - start) * 1000
        top = int(np.argmax(probs))
        return Prediction(
            classes=list(self.classes),
            probabilities=[float(p) for p in probs],
            label=self.classes[top],
            confidence=float(probs[top]),
            abstained=bool(probs[top] < abstain_below),
            latency_ms=latency,
            ood_score=ood_score,
            in_distribution=in_distribution,
        )

    def explain(self, image: Image.Image, class_index: int | None = None) -> tuple[Image.Image, Image.Image]:
        """(model view, Grad-CAM overlay) for the predicted or the requested class."""
        cam, _ = self._cam(self._tensor(image), class_index)
        view = self.model_view(image)
        return view, overlay(view, cam)
