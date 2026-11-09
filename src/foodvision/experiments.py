"""The two experimental protocols, kept strictly separate.

A. **Assignment protocol** (reproduces the course assignment): nine runs (3 backbones x 3 epoch budgets), each
   logging train and test loss/accuracy to TensorBoard every epoch. The best run is chosen on the test set,
   exactly as the assignment asks, which makes its score optimistically biased.
B. **Research protocol**: a stratified validation split of the training images drives every decision
   (backbone, epoch budget, temperature). The test set is used once for the selected model; comparisons with
   the other backbones afterwards are labelled exploratory.
"""

from __future__ import annotations

import logging
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF

from . import analysis as A
from . import data as D
from . import ood as O
from .calibration import calibration_summary, softmax
from .config import Config
from .engine import evaluate, fit
from .explain import GradCAM, overlay
from .metrics import classification_report, macro_f1
from .models import REGISTRY, build_model, get_transforms, parameter_counts
from .stats import bootstrap_ci, holm_correction, mcnemar_exact, paired_bootstrap_difference, wilson_interval

LOG = logging.getLogger(__name__)


@dataclass
class DataBundle:
    root: Path
    classes: list[str]
    train: list[D.Sample]
    test: list[D.Sample]


def prepare_data(cfg: Config) -> DataBundle:
    """Download (once) and index the dataset; the class mapping must match across splits."""
    root = D.download_dataset(cfg.data.root, cfg.data.name, cfg.data.urls)
    classes = D.class_names(root / "train")
    if classes != D.class_names(root / "test"):
        raise ValueError("Train and test splits have different class folders")
    return DataBundle(root, classes, D.index_split(root / "train", classes), D.index_split(root / "test", classes))


def _writer(path: Path) -> Any:
    from torch.utils.tensorboard import SummaryWriter

    shutil.rmtree(path, ignore_errors=True)  # a re-run replaces old curves instead of overlaying them
    return SummaryWriter(log_dir=str(path))


def _total_params(name: str, n_classes: int) -> int:
    return parameter_counts(build_model(name, n_classes, pretrained=False))["total"]


# --------------------------------------------------------------------------- A. assignment protocol
def run_assignment(cfg: Config, bundle: DataBundle, device: torch.device) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Nine independent runs with train/test logging; returns the summary table and full histories."""
    k, bs = len(bundle.classes), cfg.train.batch_size
    rows, histories = [], {}
    for name in cfg.models:
        tf = get_transforms(name)
        xtr, ytr = D.load_tensors(bundle.train, tf)
        xte, yte = D.load_tensors(bundle.test, tf)
        for epochs in cfg.epochs:
            D.seed_everything(cfg.seed)
            model = build_model(name, k)
            loaders = {
                "train": D.make_loader(xtr, ytr, bs, True, cfg.seed, cfg.train.drop_last),
                "test": D.make_loader(xte, yte, bs, False),
            }
            writer = _writer(Path(cfg.paths.runs) / "assignment" / name / f"{epochs}_epochs")
            start = time.perf_counter()
            history, _ = fit(
                model, loaders, epochs, cfg.train.learning_rate, device, ("test",), writer,
                freeze_bn_stats=cfg.train.freeze_bn_stats, label=f"[assignment] {name} {epochs}ep",
            )
            seconds = time.perf_counter() - start
            histories[f"{name}_{epochs}"] = history
            n_test = len(yte)
            correct = round(history["test"]["acc"][-1] * n_test)
            low, high = wilson_interval(correct, n_test)
            counts = parameter_counts(model)
            rows.append(
                {
                    "model": name,
                    "epochs": epochs,
                    "train_loss": history["train"]["loss"][-1],
                    "train_acc": history["train"]["acc"][-1],
                    "test_loss": history["test"]["loss"][-1],
                    "test_acc": history["test"]["acc"][-1],
                    "test_correct": correct,
                    "n_test": n_test,
                    "test_ci_low": low,
                    "test_ci_high": high,
                    "train_seconds": seconds,
                    "seconds_per_epoch": seconds / epochs,
                    "total_params": counts["total"],
                    "trainable_params": counts["trainable"],
                }
            )
        del xtr, xte
    frame = pd.DataFrame(rows)
    best = frame.sort_values(["test_acc", "test_loss"], ascending=[False, True]).iloc[0]
    frame["selected"] = (frame["model"] == best["model"]) & (frame["epochs"] == best["epochs"])
    return frame, histories


# --------------------------------------------------------------------------- B. research protocol
@dataclass
class ResearchResult:
    validation: pd.DataFrame
    histories: dict[str, Any]
    selection: dict[str, Any]
    test_report: dict[str, Any]
    predictions: pd.DataFrame
    calibration: dict[str, Any]
    bn_ablation: pd.DataFrame | None
    model: nn.Module
    model_name: str
    epochs: int
    test_samples: list[D.Sample]
    test_probs: np.ndarray
    ood: dict[str, Any]
    ood_scores: pd.DataFrame | None


def _features_and_logits(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled penultimate features and logits in one pass; the same computation as ``model(x)`` for EfficientNet."""
    features = torch.flatten(model.avgpool(model.features(x)), 1)
    return features, model.classifier(features)


@torch.inference_mode()
def features_for_files(
    model: nn.Module, paths: list[Path], transform: Any, device: torch.device, batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Features and logits for image files, decoded batch by batch; unreadable files are skipped (left out of kept)."""
    model.eval()
    kept: list[int] = []
    features: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    batch: list[torch.Tensor] = []

    def flush() -> None:
        if batch:
            feats, logits = _features_and_logits(model, torch.stack(batch).to(device))
            features.append(feats.float().cpu())
            outputs.append(logits.float().cpu())
            batch.clear()

    for index, path in enumerate(paths):
        try:
            with Image.open(path) as image:
                batch.append(transform(image.convert("RGB")))
        except (OSError, ValueError) as error:  # corrupt or truncated file in a public dataset
            LOG.warning("Skipping unreadable image %s: %s", path, error)
            continue
        kept.append(index)
        if len(batch) == batch_size:
            flush()
    flush()
    if not outputs:
        return np.empty((0, 0)), np.empty((0, 0)), kept
    return torch.cat(features).numpy(), torch.cat(outputs).numpy(), kept


def logits_for_files(
    model: nn.Module, paths: list[Path], transform: Any, device: torch.device, batch_size: int = 32
) -> tuple[np.ndarray, list[int]]:
    """Logits for image files; unreadable files are skipped and their indices left out."""
    _, logits, kept = features_for_files(model, paths, transform, device, batch_size)
    return logits, kept


def not_food_check(
    cfg: Config, model: nn.Module, name: str, val_logits: np.ndarray, test_logits: np.ndarray, temperature: float,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Choose an out-of-distribution score and threshold on validation data and test them once (see ``ood``)."""
    root = Path(cfg.data.root) / "ood"
    files = {source: D.download_ood_source(root, source) for source in cfg.ood.sources}
    files = {source: paths for source, paths in files.items() if paths}
    if not files:
        reason = "none of the non-food photo collections could be downloaded"
        LOG.warning("Not-food check skipped: %s. Re-run with --stages research --force to retry.", reason)
        return {"status": "skipped", "reason": reason}, None
    val_pairs, test_pairs = O.sample_split(files, cfg.ood.per_source, cfg.seed)
    tf = get_transforms(name)
    val_ood, kept_val = logits_for_files(model, [p for p, _ in val_pairs], tf, device, cfg.train.batch_size)
    test_ood, kept_test = logits_for_files(model, [p for p, _ in test_pairs], tf, device, cfg.train.batch_size)
    if len(kept_val) < 2 or len(kept_test) < 2:
        return {"status": "skipped", "reason": "too few readable non-food photos"}, None
    test_sources = [test_pairs[i][1] for i in kept_test]
    report = O.evaluate(
        val_logits, val_ood, test_logits, test_ood, test_sources, temperature, cfg.ood.target_tpr,
        cfg.evaluation.bootstrap_resamples, cfg.seed,
    )
    report["sources"] = {
        source: {**O.SOURCES[source], "available": len(paths)} for source, paths in sorted(files.items())
    }
    table = O.score_table(
        {"val_food": val_logits, "val_non_food": val_ood, "test_food": test_logits, "test_non_food": test_ood},
        {"val_food": ["foodvision"] * len(val_logits), "val_non_food": [val_pairs[i][1] for i in kept_val],
         "test_food": ["foodvision"] * len(test_logits), "test_non_food": test_sources},
        temperature,
    )
    test = report["test"]
    LOG.info(
        "Not-food check (%s): test AUROC %.3f; rejects %.0f%% of non-food photos, accepts %.0f%% of food photos",
        report["score"], test["auroc"]["estimate"], 100 * test["rejected_non_food"], 100 * test["accepted_food"],
    )
    return report, table


def run_research(cfg: Config, bundle: DataBundle, device: torch.device) -> ResearchResult:
    """Validation-based selection over seeds, one-time test evaluation, calibration and exploratory tests."""
    k, bs, classes = len(bundle.classes), cfg.train.batch_size, bundle.classes
    train_s, val_s = D.stratified_split(bundle.train, cfg.data.val_fraction, cfg.seed)
    budgets = sorted(cfg.epochs)
    val_rows: list[dict[str, Any]] = []
    bn_rows: list[dict[str, Any]] = []
    histories: dict[str, Any] = {}
    snapshots: dict[tuple[int, str, int], dict[str, torch.Tensor]] = {}
    val_logits: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    test_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    for name in cfg.models:
        tf = get_transforms(name)
        xtr, ytr = D.load_tensors(train_s, tf)
        xva, yva = D.load_tensors(val_s, tf)
        test_cache[name] = D.load_tensors(bundle.test, tf)
        val_loader = D.make_loader(xva, yva, bs, False)
        for seed in cfg.research.seeds:
            D.seed_everything(seed)
            model = build_model(name, k)
            loaders = {"train": D.make_loader(xtr, ytr, bs, True, seed, cfg.train.drop_last), "val": val_loader}
            writer = _writer(Path(cfg.paths.runs) / "research" / f"seed_{seed}" / name)
            history, snaps = fit(
                model, loaders, max(budgets), cfg.train.learning_rate, device, ("val",), writer, budgets,
                cfg.train.freeze_bn_stats, label=f"[research] {name} seed {seed}",
            )
            histories[f"{name}_seed{seed}"] = history
            for epochs in budgets:  # the budgets are stopping points on one run
                model.load_state_dict(snaps[epochs], strict=False)
                result = evaluate(model, val_loader, device)
                preds = result["logits"].argmax(1)
                val_rows.append(
                    {
                        "seed": seed,
                        "model": name,
                        "epochs": epochs,
                        "val_loss": result["loss"],
                        "val_acc": result["acc"],
                        "val_macro_f1": macro_f1(result["labels"], preds, k),
                    }
                )
                snapshots[(seed, name, epochs)] = snaps[epochs]
                if seed == cfg.seed:
                    val_logits[(name, epochs)] = (result["logits"], result["labels"])
        if cfg.research.bn_ablation:
            frozen = not cfg.train.freeze_bn_stats
            D.seed_everything(cfg.seed)
            model = build_model(name, k)
            loaders = {"train": D.make_loader(xtr, ytr, bs, True, cfg.seed, cfg.train.drop_last), "val": val_loader}
            _, snaps = fit(
                model, loaders, max(budgets), cfg.train.learning_rate, device, ("val",), None, budgets, frozen,
                label=f"[bn ablation] {name}",
            )
            for epochs in budgets:
                model.load_state_dict(snaps[epochs], strict=False)
                result = evaluate(model, val_loader, device)
                bn_rows.append(
                    {"model": name, "epochs": epochs, "bn_statistics": "frozen" if frozen else "adapting",
                     "val_acc": result["acc"], "val_loss": result["loss"]}
                )
        del xtr, xva

    validation = pd.DataFrame(val_rows)
    params = {name: _total_params(name, k) for name in cfg.models}
    table = (
        validation.groupby(["model", "epochs"])
        .agg(
            val_acc_mean=("val_acc", "mean"),
            val_acc_sd=("val_acc", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_macro_f1_mean=("val_macro_f1", "mean"),
        )
        .reset_index()
    )
    table["total_params"] = table["model"].map(params)
    table = table.sort_values(
        ["val_acc_mean", "val_loss_mean", "total_params", "epochs"], ascending=[False, True, True, True]
    ).reset_index(drop=True)
    name, epochs = str(table.loc[0, "model"]), int(table.loc[0, "epochs"])
    selection = {
        "rule": "highest mean validation accuracy over seeds; ties: lower validation loss, fewer parameters, fewer epochs",
        "model": name,
        "label": REGISTRY[name].label,
        "epochs": epochs,
        "split": {"train": len(train_s), "validation": len(val_s), "test": len(bundle.test)},
        "seeds": list(cfg.research.seeds),
        "table": table.to_dict("records"),
    }
    LOG.info("Selected %s at %d epochs on validation data", name, epochs)

    # ---- one-time test evaluation of the selected configuration (seed = cfg.seed)
    model = build_model(name, k).to(device)
    model.load_state_dict(snapshots[(cfg.seed, name, epochs)], strict=False)
    xte, yte = test_cache[name]
    test_loader = D.make_loader(xte, yte, bs, False)
    result = evaluate(model, test_loader, device)
    logits, labels = result["logits"], result["labels"]
    calibration = calibration_summary(*val_logits[(name, epochs)], logits, labels, cfg.evaluation.calibration_bins)
    probs = softmax(logits, calibration["temperature"])
    preds = probs.argmax(1)
    correct = preds == labels
    report = classification_report(labels, preds, classes)
    ood_report: dict[str, Any] = {"status": "disabled"}
    ood_scores = None
    if cfg.ood.enabled:
        ood_report, ood_scores = not_food_check(
            cfg, model, name, val_logits[(name, epochs)][0], logits, calibration["temperature"], device
        )
    n_boot = cfg.evaluation.bootstrap_resamples
    low, high = wilson_interval(int(correct.sum()), len(labels))

    # Repeatability: the same selected configuration trained with the other seeds.
    repeat = []
    for seed in cfg.research.seeds:
        model.load_state_dict(snapshots[(seed, name, epochs)], strict=False)
        r = evaluate(model, test_loader, device)
        repeat.append({"seed": seed, "test_acc": r["acc"], "test_loss": r["loss"]})
    model.load_state_dict(snapshots[(cfg.seed, name, epochs)], strict=False)

    # Exploratory, post-selection: the other backbones at the same budget on the same test images.
    comparisons = []
    for other in cfg.models:
        if other == name:
            continue
        rival = build_model(other, k).to(device)
        rival.load_state_dict(snapshots[(cfg.seed, other, epochs)], strict=False)
        x2, y2 = test_cache[other]
        r2 = evaluate(rival, D.make_loader(x2, y2, bs, False), device)
        if not np.array_equal(r2["labels"], labels):
            raise RuntimeError("Test order differs between models; paired tests would be invalid")
        rival_correct = r2["logits"].argmax(1) == labels
        comparisons.append(
            {
                "model_a": name,
                "model_b": other,
                "epochs": epochs,
                "acc_a": float(correct.mean()),
                "acc_b": float(rival_correct.mean()),
                **mcnemar_exact(correct, rival_correct),
                "difference": paired_bootstrap_difference(correct, rival_correct, n_boot, seed=cfg.seed),
            }
        )
    for comparison, adjusted in zip(comparisons, holm_correction([c["p_value"] for c in comparisons])):
        comparison["p_holm"] = adjusted

    test_report = {
        "model": name,
        "label": REGISTRY[name].label,
        "epochs": epochs,
        "seed": cfg.seed,
        "n_test": int(len(labels)),
        "n_correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "accuracy_wilson": [low, high],
        "accuracy_bootstrap": bootstrap_ci(lambda c: float(c.mean()), [correct.astype(float)], n_boot, seed=cfg.seed),
        "macro_f1": report["macro_f1"],
        "macro_f1_bootstrap": bootstrap_ci(lambda t, p: macro_f1(t, p, k), [labels, preds], n_boot, seed=cfg.seed),
        "weighted_f1": report["weighted_f1"],
        "loss": result["loss"],
        "per_class": report["per_class"],
        "confusion_matrix": report["confusion_matrix"],
        "classes": classes,
        "repeatability": repeat,
        "comparisons": comparisons,
        "protocol_note": "Selected on validation data; first and only confirmatory use of the test set. "
        "Comparisons with other backbones were run after selection and are exploratory.",
    }
    raw = softmax(logits)
    predictions = pd.DataFrame(
        {
            "path": [str(s.path.relative_to(bundle.root)) for s in bundle.test],
            "label": [classes[i] for i in labels],
            "prediction": [classes[i] for i in preds],
            "confidence": probs.max(1),
            "correct": correct,
            **{f"p_{c}": probs[:, i] for i, c in enumerate(classes)},
            **{f"raw_p_{c}": raw[:, i] for i, c in enumerate(classes)},
        }
    )
    bn_ablation = None
    if bn_rows:
        baseline = validation[validation["seed"] == cfg.seed].assign(
            bn_statistics="frozen" if cfg.train.freeze_bn_stats else "adapting"
        )[["model", "epochs", "bn_statistics", "val_acc", "val_loss"]]
        bn_ablation = pd.concat([baseline, pd.DataFrame(bn_rows)], ignore_index=True)
    return ResearchResult(
        validation, histories, selection, test_report, predictions, calibration, bn_ablation,
        model, name, epochs, list(bundle.test), probs, ood_report, ood_scores,
    )


# --------------------------------------------------------------------------- explainability and cost
def make_gallery(result: ResearchResult, out_dir: Path, size: int, thumb: int = 260) -> list[dict[str, Any]]:
    """Grad-CAM thumbnails: every misclassified test image, then confident and borderline correct ones."""
    frame = result.predictions
    wrong = frame.index[~frame["correct"]].tolist()
    right = frame[frame["correct"]].sort_values("confidence")
    confident = right.groupby("label", sort=True)["confidence"].idxmax().tolist()
    borderline = right.index[:3].tolist()
    chosen = list(dict.fromkeys(wrong[: max(1, size // 2)] + confident + borderline))[:size]
    model = result.model
    cam = GradCAM(model)
    tf = get_transforms(result.model_name)
    device = next(model.parameters()).device
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, idx in enumerate(chosen):
        row = frame.loc[idx]
        with Image.open(result.test_samples[idx].path) as raw_image:
            image = raw_image.convert("RGB")
        view = TF.center_crop(TF.resize(image, tf.resize_size, interpolation=tf.interpolation), tf.crop_size)
        heat, _ = cam(tf(image).unsqueeze(0).to(device))
        stem = f"{i:02d}_{row['label']}_as_{row['prediction']}"
        view.resize((thumb, thumb)).save(out_dir / f"{stem}_view.jpg", quality=85)
        overlay(view, heat).resize((thumb, thumb)).save(out_dir / f"{stem}_cam.jpg", quality=85)
        kind = "misclassified" if not row["correct"] else "most confident" if idx in confident else "least confident"
        entries.append(
            {
                "view": f"{stem}_view.jpg",
                "cam": f"{stem}_cam.jpg",
                "label": row["label"],
                "prediction": row["prediction"],
                "confidence": float(row["confidence"]),
                "correct": bool(row["correct"]),
                "kind": kind,
            }
        )
    return entries


@torch.inference_mode()
def benchmark_latency(models: tuple[str, ...], runs: int) -> list[dict[str, Any]]:
    """Median single-image latency on CPU (what the deployed app uses) and on GPU when available."""
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = []
    for name in models:
        size = get_transforms(name).crop_size[0]
        model = build_model(name, 3, pretrained=False).eval()
        for device in devices:
            model.to(device)
            x = torch.randn(1, 3, size, size, device=device)
            for _ in range(3):  # warm-up
                model(x)
            times = []
            for _ in range(runs):
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            rows.append(
                {
                    "model": name,
                    "device": device,
                    "input": size,
                    "median_ms": statistics.median(times),
                    "p90_ms": float(np.percentile(times, 90)),
                    "threads": torch.get_num_threads(),
                    "runs": runs,
                }
            )
    return rows


# --------------------------------------------------------------------------- analysis of the selected model
@torch.inference_mode()
def robustness_eval(
    model: nn.Module, name: str, samples: list[D.Sample], temperature: float, ood_meta: dict[str, Any] | None,
    device: torch.device, seed: int, batch_size: int = 32,
) -> pd.DataFrame:
    """Accuracy on corrupted copies of the test photos and, with a not-food check, how much real food it still accepts.

    Corruptions are applied to the model's view (resized and centre-cropped), so the clean condition reproduces the
    regular test preprocessing exactly. Exploratory: no decision is made from these results.
    """
    model.eval()
    tf = get_transforms(name)
    views = []
    for sample in samples:
        with Image.open(sample.path) as raw:
            image = raw.convert("RGB")
        views.append(TF.center_crop(TF.resize(image, tf.resize_size, interpolation=tf.interpolation), tf.crop_size))
    labels = np.array([s.label for s in samples])
    rows = []
    for kind, severity in A.conditions():
        rng = np.random.default_rng(seed)  # the same noise pattern in every run
        chunks = []
        for start in range(0, len(views), batch_size):
            batch = [TF.normalize(TF.to_tensor(A.corrupt(v, kind, severity, rng)), tf.mean, tf.std)
                     for v in views[start:start + batch_size]]
            chunks.append(model(torch.stack(batch).to(device)).float().cpu())
        logits = torch.cat(chunks).numpy()
        probs = softmax(logits, temperature)
        correct = probs.argmax(1) == labels
        low, high = wilson_interval(int(correct.sum()), len(labels))
        row: dict[str, Any] = {
            "corruption": kind,
            "label": "Clean" if kind == "none" else A.CORRUPTIONS[kind]["label"],
            "severity": severity,
            "level": None if kind == "none" else A.CORRUPTIONS[kind]["levels"][severity - 1],
            "accuracy": float(correct.mean()),
            "ci_low": low,
            "ci_high": high,
            "mean_confidence": float(probs.max(1).mean()),
            "n": len(labels),
        }
        if ood_meta:
            score = O.scores(logits, temperature)[ood_meta["score"]]
            row["accepted_as_food"] = float((score >= ood_meta["threshold"]).mean())
        rows.append(row)
        if severity in (0, 3):
            LOG.info("[robustness] %s %s: accuracy %.1f%%", row["label"], "strong" if severity else "", 100 * row["accuracy"])
    return pd.DataFrame(rows)


def feature_map(
    cfg: Config, model: nn.Module, name: str, bundle: DataBundle, temperature: float,
    ood_meta: dict[str, Any] | None, device: torch.device,
) -> pd.DataFrame:
    """2D map of what the network sees: the test food photos plus the non-food test sample of the not-food check."""
    tf = get_transforms(name)
    feats, logits, kept = features_for_files(model, [s.path for s in bundle.test], tf, device, cfg.train.batch_size)
    labels = [bundle.classes[bundle.test[i].label] for i in kept]
    groups = ["food"] * len(kept)
    if cfg.ood.enabled:
        files = {source: D.download_ood_source(Path(cfg.data.root) / "ood", source) for source in cfg.ood.sources}
        files = {source: paths for source, paths in files.items() if paths}
        if files:
            _, test_pairs = O.sample_split(files, cfg.ood.per_source, cfg.seed)  # the same photos as the check
            more_feats, more_logits, more_kept = features_for_files(
                model, [p for p, _ in test_pairs], tf, device, cfg.train.batch_size
            )
            if more_kept:
                feats = np.concatenate([feats, more_feats])
                logits = np.concatenate([logits, more_logits])
                labels += [O.SOURCES[test_pairs[i][1]]["content"] for i in more_kept]
                groups += ["non-food"] * len(more_kept)
    xy = A.project_2d(feats, cfg.seed, cfg.analysis.perplexity)
    probs = softmax(logits, temperature)
    frame = pd.DataFrame(
        {"x": xy[:, 0], "y": xy[:, 1], "group": groups, "label": labels,
         "prediction": [bundle.classes[i] for i in probs.argmax(1)], "confidence": probs.max(1)}
    )
    is_food = frame["group"] == "food"
    frame["outcome"] = np.where(is_food, np.where(frame["label"] == frame["prediction"], "correct", "misclassified"),
                                "non-food")
    if ood_meta:
        frame["accepted_as_food"] = O.scores(logits, temperature)[ood_meta["score"]] >= ood_meta["threshold"]
    LOG.info("[feature map] %d food and %d non-food photos projected with t-SNE", is_food.sum(), (~is_food).sum())
    return frame


def data_efficiency(
    cfg: Config, bundle: DataBundle, selection: dict[str, Any], validation: pd.DataFrame, device: torch.device
) -> pd.DataFrame:
    """Validation accuracy of the selected configuration trained on nested fractions of the training split.

    Every fraction uses the same epochs and learning rate; the full-data points are the research runs themselves
    (same split, seeds and epochs), so they are not trained twice.
    """
    name, epochs = str(selection["model"]), int(selection["epochs"])
    k, bs = len(bundle.classes), cfg.train.batch_size
    train_s, val_s = D.stratified_split(bundle.train, cfg.data.val_fraction, cfg.seed)
    tf = get_transforms(name)
    xtr, ytr = D.load_tensors(train_s, tf)
    xva, yva = D.load_tensors(val_s, tf)
    val_loader = D.make_loader(xva, yva, bs, False)
    rows: list[dict[str, Any]] = []
    for fraction in sorted(f for f in cfg.analysis.fractions if f < 1.0):
        for seed in cfg.research.seeds:
            index = torch.from_numpy(A.nested_subset(ytr.numpy(), fraction, seed))
            D.seed_everything(seed)
            model = build_model(name, k)
            drop_last = cfg.train.drop_last and len(index) > bs
            loaders = {"train": D.make_loader(xtr[index], ytr[index], bs, True, seed, drop_last)}
            fit(model, loaders, epochs, cfg.train.learning_rate, device, (), None, (), cfg.train.freeze_bn_stats,
                label=f"[data efficiency] {name} {fraction:.0%} seed {seed}")
            result = evaluate(model, val_loader, device)
            rows.append({"fraction": fraction, "n_train": len(index), "seed": seed, "val_acc": result["acc"],
                         "val_loss": result["loss"], "source": "this analysis"})
    if 1.0 in cfg.analysis.fractions:
        full = validation[(validation["model"] == name) & (validation["epochs"] == epochs)]
        rows += [{"fraction": 1.0, "n_train": len(train_s), "seed": int(r.seed), "val_acc": float(r.val_acc),
                  "val_loss": float(r.val_loss), "source": "research runs"} for r in full.itertuples()]
    del xtr, xva
    return pd.DataFrame(rows)
