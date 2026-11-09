"""Training and evaluation loops with per-epoch TensorBoard logging and in-training snapshots."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import export_delta, set_train_mode

LOG = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    freeze_bn_stats: bool = False,
) -> tuple[float, float]:
    """One pass over the training data; returns sample-weighted (loss, accuracy)."""
    set_train_mode(model, freeze_bn_stats)
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += int((logits.argmax(dim=1) == y).sum())
        seen += len(y)
    return total_loss / seen, correct / seen


@torch.inference_mode()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Logits and labels for a whole loader, in loader order (eval mode)."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Loss, accuracy, logits and labels on one split."""
    logits, labels = predict_logits(model, loader, device)
    loss = float(nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)))
    return {"loss": loss, "acc": float((logits.argmax(1) == labels).mean()), "logits": logits, "labels": labels}


def fit(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    epochs: int,
    learning_rate: float,
    device: torch.device,
    eval_splits: Iterable[str] = ("test",),
    writer: Any | None = None,
    snapshot_epochs: Iterable[int] = (),
    freeze_bn_stats: bool = False,
    label: str = "",
) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    """Train the head with Adam and evaluate every epoch.

    Writes ``Loss/<split>`` and ``Accuracy/<split>`` to TensorBoard when a ``SummaryWriter`` is given and
    returns the history plus delta snapshots taken at ``snapshot_epochs``.
    """
    model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    eval_splits = tuple(eval_splits)
    wanted = set(snapshot_epochs)
    history: dict[str, Any] = {split: {"loss": [], "acc": []} for split in ("train", *eval_splits)}
    history["seconds"] = []
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        loss, acc = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device, freeze_bn_stats)
        history["train"]["loss"].append(loss)
        history["train"]["acc"].append(acc)
        for split in eval_splits:
            result = evaluate(model, loaders[split], device)
            history[split]["loss"].append(result["loss"])
            history[split]["acc"].append(result["acc"])
        history["seconds"].append(time.perf_counter() - start)
        if writer is not None:
            for split in ("train", *eval_splits):
                writer.add_scalar(f"Loss/{split}", history[split]["loss"][-1], epoch)
                writer.add_scalar(f"Accuracy/{split}", history[split]["acc"][-1], epoch)
            writer.flush()
        if epoch in wanted:
            snapshots[epoch] = export_delta(model)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            scores = " | ".join(f"{s} {history[s]['loss'][-1]:.3f}/{history[s]['acc'][-1]:.1%}" for s in ("train", *eval_splits))
            LOG.info("%s epoch %2d/%d | %s", label, epoch, epochs, scores)
    if writer is not None:
        writer.close()
    return history, snapshots
