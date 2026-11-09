"""Grad-CAM (Selvaraju et al., 2017) for a frozen-backbone classifier.

Grad-CAM highlights image regions whose last-layer activations most increase the chosen class score.
It is a correlational visual summary of the model's evidence, not a causal explanation.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from torch import nn


class GradCAM:
    """Class-activation maps from the final convolutional block (``model.features[-1]`` by default)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.layer = target_layer if target_layer is not None else model.features[-1]
        self._lock = threading.Lock()  # hooks are per-call; serialise concurrent app sessions

    def __call__(self, x: torch.Tensor, class_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (cam in [0, 1] at feature-map resolution, softmax probabilities) for a 1-image batch."""
        store: dict[str, torch.Tensor] = {}

        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
            # The backbone is frozen, so its output does not track gradients; make it a leaf that does.
            leaf = output.detach().requires_grad_(True)
            store["activations"] = leaf
            return leaf

        with self._lock:
            handle = self.layer.register_forward_hook(hook)
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.enable_grad():
                    logits = self.model(x)
                    index = int(logits[0].argmax()) if class_index is None else int(class_index)
                    self.model.zero_grad(set_to_none=True)
                    logits[0, index].backward()
                activations = store["activations"]
                gradients = activations.grad
                if gradients is None:
                    raise RuntimeError("Grad-CAM received no gradients for the target layer")
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activations).sum(dim=1))[0]
                peak = cam.max()
                cam = cam / peak if peak > 0 else cam
                probs = torch.softmax(logits.detach(), dim=1)[0]
            finally:
                handle.remove()
                self.model.train(was_training)
        return cam.detach().cpu().numpy(), probs.cpu().numpy()


def upsample(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a [0, 1] map to (width, height)."""
    image = Image.fromarray(np.uint8(np.clip(cam, 0, 1) * 255))
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=float) / 255.0


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45, cmap: str = "inferno") -> Image.Image:
    """Blend a heat map over an RGB image of the same view the model saw."""
    base = np.asarray(image.convert("RGB"), dtype=float) / 255.0
    heat = colormaps[cmap](upsample(cam, image.size))[..., :3]
    blended = (1 - alpha) * base + alpha * heat
    return Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255))
