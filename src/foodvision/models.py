"""EfficientNet-B2/B3/B4 for feature-extraction transfer learning, with compact *delta* checkpoints.

A delta checkpoint stores only what training changes: the new classifier head and the BatchNorm running
statistics (which keep updating in train mode even when the backbone parameters are frozen). The frozen
convolution weights are re-created from the pinned torchvision ImageNet weights, so a trained model fits in
well under a megabyte and can be versioned in Git.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch import nn

from .artifacts import to_builtin

CHECKPOINT_FORMAT = "foodvision-delta/1"
_BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


@dataclass(frozen=True)
class Backbone:
    builder: Any
    weights: Any
    label: str


REGISTRY: dict[str, Backbone] = {
    "effnetb2": Backbone(
        torchvision.models.efficientnet_b2, torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1, "EfficientNet-B2"
    ),
    "effnetb3": Backbone(
        torchvision.models.efficientnet_b3, torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1, "EfficientNet-B3"
    ),
    "effnetb4": Backbone(
        torchvision.models.efficientnet_b4, torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1, "EfficientNet-B4"
    ),
}


def _spec(name: str) -> Backbone:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name]


def get_transforms(name: str) -> Any:
    """The exact preprocessing the pinned ImageNet weights were evaluated with."""
    return _spec(name).weights.transforms()


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """ImageNet EfficientNet with a frozen feature extractor and a new ``Dropout -> Linear`` head."""
    spec = _spec(name)
    model = spec.builder(weights=spec.weights if pretrained else None)
    for param in model.features.parameters():
        param.requires_grad_(False)
    dropout = model.classifier[0].p
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(in_features, num_classes))
    return model


def set_train_mode(model: nn.Module, freeze_bn_stats: bool = False) -> None:
    """Training mode for the head; optionally keep the backbone in eval mode.

    ``requires_grad=False`` stops weight updates but *not* BatchNorm running-statistic updates. With
    ``freeze_bn_stats=True`` the backbone stays in eval mode, so BatchNorm keeps its ImageNet statistics
    (and stochastic depth is disabled).
    """
    model.train()
    if freeze_bn_stats:
        model.features.eval()


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def export_delta(model: nn.Module) -> dict[str, torch.Tensor]:
    """The state that training changed: classifier tensors and BatchNorm running statistics."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith("classifier.") or key.endswith(_BN_BUFFERS)
    }


def save_checkpoint(
    model: nn.Module, path: str | Path, name: str, classes: list[str], meta: dict[str, Any]
) -> dict[str, Any]:
    """Write a self-describing delta checkpoint and return its size and SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "arch": name,
        "weights": _spec(name).weights.name,
        "classes": list(classes),
        "delta": export_delta(model),
        "meta": to_builtin(meta),  # NumPy numbers would make torch.load(weights_only=True) refuse the file
    }
    torch.save(payload, path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_checkpoint(path: str | Path, pretrained: bool = True, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the trained model from pinned ImageNet weights plus the stored delta (eval mode)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if payload["weights"] != _spec(payload["arch"]).weights.name:
        raise ValueError("Checkpoint was trained on different pretrained weights than this code pins")
    model = build_model(payload["arch"], len(payload["classes"]), pretrained=pretrained)
    result = model.load_state_dict(payload["delta"], strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected keys in checkpoint: {result.unexpected_keys[:5]}")
    return model.to(device).eval(), payload


@torch.no_grad()
def architecture_summary(name: str) -> dict[str, Any]:
    """Stage-level structure (blocks, output shape, parameters) measured with forward hooks."""
    spec = _spec(name)
    tf = spec.weights.transforms()
    model = build_model(name, 3, pretrained=False).eval()
    shapes: dict[int, list[int]] = {}
    hooks = [
        stage.register_forward_hook(lambda _m, _i, out, k=k: shapes.__setitem__(k, list(out.shape[1:])))
        for k, stage in enumerate(model.features)
    ]
    size = tf.crop_size[0]
    model(torch.zeros(1, 3, size, size))
    for hook in hooks:
        hook.remove()
    last = len(model.features) - 1
    stages = [
        {
            "stage": k,
            "kind": "stem" if k == 0 else "head conv" if k == last else "MBConv stage",
            "blocks": len(stage) if 0 < k < last else 1,
            "output": shapes[k],
            "params": sum(p.numel() for p in stage.parameters()),
        }
        for k, stage in enumerate(model.features)
    ]
    original = spec.builder(weights=None)
    return {
        "name": name,
        "label": spec.label,
        "weights": spec.weights.name,
        "resize": tf.resize_size[0],
        "crop": size,
        "interpolation": str(tf.interpolation.value),
        "mean": list(tf.mean),
        "std": list(tf.std),
        "mbconv_blocks": sum(s["blocks"] for s in stages if s["kind"] == "MBConv stage"),
        "feature_dim": model.classifier[1].in_features,
        "dropout": model.classifier[0].p,
        "params_imagenet_head": sum(p.numel() for p in original.parameters()),
        "params": parameter_counts(model),
        "imagenet_top1": spec.weights.meta.get("_metrics", {}).get("ImageNet-1K", {}).get("acc@1"),
        "gflops": spec.weights.meta.get("_ops"),
        "stages": stages,
    }
