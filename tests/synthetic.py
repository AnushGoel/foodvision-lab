"""Synthetic, clearly fake artifacts and images for tests.

They follow the real artifact schema and are computed with the real statistics code, so every page and report
can be exercised in CI without a GPU, a dataset download or a trained model. They are never used by the app.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from foodvision.artifacts import SCHEMA_VERSION, ArtifactStore, Files
from foodvision.analysis import CORRUPTIONS, conditions
from foodvision.audit import audit_dataset
from foodvision.calibration import calibration_summary, softmax
from foodvision.metrics import classification_report, macro_f1
from foodvision.ood import SOURCES, evaluate, score_table
from foodvision.figures import summary_figure
from foodvision.reports import write_reports
from foodvision.stats import bootstrap_ci, holm_correction, mcnemar_exact, paired_bootstrap_difference, wilson_interval

CLASSES = ["pizza", "steak", "sushi"]
MODELS = ["effnetb2", "effnetb3", "effnetb4"]
EPOCHS = [15, 20, 30]
COLORS = {"pizza": (200, 70, 50), "steak": (120, 75, 50), "sushi": (60, 120, 100)}


def make_image_folder(root: Path, per_class: tuple[int, int] = (6, 3), seed: int = 0) -> Path:
    """ImageFolder-style train/test tree of random-shape JPEGs with varied sizes."""
    rng = np.random.default_rng(seed)
    for split, n in zip(("train", "test"), per_class):
        for cls in CLASSES:
            folder = root / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                w, h = int(rng.integers(160, 420)), int(rng.integers(160, 420))
                image = Image.new("RGB", (w, h), COLORS[cls])
                draw = ImageDraw.Draw(image)
                for _ in range(6):
                    x, y, r = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(8, 50))
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
                image.save(folder / f"{cls}_{i}.jpg", quality=90)
    return root


def _logits(labels: np.ndarray, accuracy: float, rng: np.random.Generator, scale: float = 4.0) -> np.ndarray:
    """Fake logits whose argmax matches the label with roughly the requested accuracy."""
    n = len(labels)
    target = np.where(rng.random(n) < accuracy, labels, (labels + rng.integers(1, 3, n)) % 3)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), target] += scale
    return logits


def _history(epochs: int, splits: tuple[str, str], rng: np.random.Generator, ceiling: float) -> dict:
    t = np.arange(1, epochs + 1)
    out: dict = {"seconds": list(5 + rng.normal(0, 0.3, epochs))}
    for split in splits:
        out[split] = {
            "loss": list(0.18 + 0.85 * np.exp(-t / 4) + np.abs(rng.normal(0, 0.01, epochs))),
            "acc": list(np.clip(ceiling - 0.3 * np.exp(-t / 3) + rng.normal(0, 0.01, epochs), 0, 1)),
        }
    return out


def make_artifacts(root: Path, data_root: Path, seed: int = 0) -> ArtifactStore:
    """Write a complete, schema-valid synthetic artifact set (model metadata, but no checkpoint file)."""
    rng = np.random.default_rng(seed)
    store = ArtifactStore(root)
    summary, frame = audit_dataset(data_root)
    store.write_json(Files.AUDIT, summary)
    store.write_csv(Files.IMAGES, frame)
    store.write_json(
        Files.ARCHITECTURE,
        [
            {"name": m, "label": f"EfficientNet-B{m[-1]}", "weights": "IMAGENET1K_V1", "resize": r, "crop": c,
             "interpolation": "bicubic", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
             "mbconv_blocks": b, "feature_dim": f, "dropout": d, "params_imagenet_head": p + f * 1000,
             "params": {"total": p, "trainable": f * 3 + 3}, "imagenet_top1": acc, "gflops": g,
             "stages": [{"stage": 0, "kind": "stem", "blocks": 1, "output": [32, 144, 144], "params": 928}]}
            for m, r, c, b, f, d, p, acc, g in [
                ("effnetb2", 288, 288, 23, 1408, 0.3, 7_705_221, 80.6, 1.09),
                ("effnetb3", 320, 300, 26, 1536, 0.3, 10_700_843, 82.0, 1.83),
                ("effnetb4", 384, 380, 32, 1792, 0.4, 17_553_995, 83.4, 4.39),
            ]
        ],
    )
    store.write_json(
        Files.BENCHMARK,
        [{"model": m, "device": dev, "input": 300, "median_ms": ms, "p90_ms": ms * 1.1, "threads": 2, "runs": 30}
         for m, base in zip(MODELS, (60.0, 85.0, 190.0)) for dev, ms in (("cpu", base), ("cuda", base / 12))],
    )

    rows, histories = [], {}
    for m, ceiling in zip(MODELS, (0.95, 0.97, 0.95)):
        for e in EPOCHS:
            history = _history(e, ("train", "test"), rng, ceiling)
            histories[f"{m}_{e}"] = history
            correct = int(round(history["test"]["acc"][-1] * 150))
            low, high = wilson_interval(correct, 150)
            rows.append({"model": m, "epochs": e, "train_loss": history["train"]["loss"][-1],
                         "train_acc": history["train"]["acc"][-1], "test_loss": history["test"]["loss"][-1],
                         "test_acc": correct / 150, "test_correct": correct, "n_test": 150, "test_ci_low": low,
                         "test_ci_high": high, "train_seconds": 5.0 * e, "seconds_per_epoch": 5.0,
                         "total_params": 10_000_000, "trainable_params": 4611})
    assignment = pd.DataFrame(rows)
    best = assignment.sort_values(["test_acc", "test_loss"], ascending=[False, True]).index[0]
    assignment["selected"] = assignment.index == best
    store.write_csv(Files.ASSIGNMENT_SUMMARY, assignment)
    store.write_json(Files.ASSIGNMENT_HISTORY, histories)

    val_labels = np.repeat([0, 1, 2], 30)
    val_rows, research_hist = [], {}
    for m, acc in zip(MODELS, (0.93, 0.96, 0.94)):
        for s in (42, 43, 44):
            research_hist[f"{m}_seed{s}"] = _history(30, ("train", "val"), rng, acc)
            for e in EPOCHS:
                logits = _logits(val_labels, acc, rng)
                preds = logits.argmax(1)
                val_rows.append({"seed": s, "model": m, "epochs": e, "val_loss": 0.2, "val_acc": float((preds == val_labels).mean()),
                                 "val_macro_f1": macro_f1(val_labels, preds, 3)})
    validation = pd.DataFrame(val_rows)
    store.write_csv(Files.VALIDATION, validation)
    store.write_json(Files.RESEARCH_HISTORY, research_hist)
    table = (validation.groupby(["model", "epochs"]).agg(val_acc_mean=("val_acc", "mean"), val_acc_sd=("val_acc", "std"),
             val_loss_mean=("val_loss", "mean"), val_macro_f1_mean=("val_macro_f1", "mean")).reset_index()
             .sort_values(["val_acc_mean", "val_loss_mean"], ascending=[False, True]).reset_index(drop=True))
    table["total_params"] = 10_000_000
    name, epochs = str(table.loc[0, "model"]), int(table.loc[0, "epochs"])
    store.write_json(Files.SELECTION, {"rule": "synthetic", "model": name, "label": f"EfficientNet-B{name[-1]}",
                                       "epochs": epochs, "split": {"train": 360, "validation": 90, "test": 150},
                                       "seeds": [42, 43, 44], "table": table.to_dict("records")})

    test_labels = np.repeat([0, 1, 2], 50)
    test_logits = _logits(test_labels, 0.96, rng, scale=6.0)
    val_logits = _logits(val_labels, 0.96, rng, scale=6.0)
    calibration = calibration_summary(val_logits, val_labels, test_logits, test_labels, 10)
    store.write_json(Files.CALIBRATION, calibration)
    probs = softmax(test_logits, calibration["temperature"])
    preds = probs.argmax(1)
    correct = preds == test_labels
    report = classification_report(test_labels, preds, CLASSES)
    comparisons = []
    for other in [m for m in MODELS if m != name]:
        rival = _logits(test_labels, 0.93, rng).argmax(1) == test_labels
        comparisons.append({"model_a": name, "model_b": other, "epochs": epochs, "acc_a": float(correct.mean()),
                            "acc_b": float(rival.mean()), **mcnemar_exact(correct, rival),
                            "difference": paired_bootstrap_difference(correct, rival, 500)})
    for c, p in zip(comparisons, holm_correction([c["p_value"] for c in comparisons])):
        c["p_holm"] = p
    store.write_json(Files.TEST_REPORT, {
        "model": name, "label": f"EfficientNet-B{name[-1]}", "epochs": epochs, "seed": 42, "n_test": 150,
        "n_correct": int(correct.sum()), "accuracy": float(correct.mean()),
        "accuracy_wilson": list(wilson_interval(int(correct.sum()), 150)),
        "accuracy_bootstrap": bootstrap_ci(np.mean, [correct.astype(float)], 500),
        "macro_f1": report["macro_f1"], "macro_f1_bootstrap": bootstrap_ci(lambda t, p: macro_f1(t, p, 3), [test_labels, preds], 500),
        "weighted_f1": report["weighted_f1"], "loss": 0.15, "per_class": report["per_class"],
        "confusion_matrix": report["confusion_matrix"], "classes": CLASSES,
        "repeatability": [{"seed": s, "test_acc": float(correct.mean()), "test_loss": 0.15} for s in (42, 43, 44)],
        "comparisons": comparisons, "protocol_note": "synthetic",
    })
    store.write_csv(Files.PREDICTIONS, pd.DataFrame({
        "path": [f"test/{CLASSES[y]}/{i}.jpg" for i, y in enumerate(test_labels)],
        "label": [CLASSES[y] for y in test_labels], "prediction": [CLASSES[y] for y in preds],
        "confidence": probs.max(1), "correct": correct,
        **{f"p_{c}": probs[:, i] for i, c in enumerate(CLASSES)},
        **{f"raw_p_{c}": softmax(test_logits)[:, i] for i, c in enumerate(CLASSES)},
    }))
    store.write_csv(Files.BN_ABLATION, pd.DataFrame(
        [{"model": m, "epochs": e, "bn_statistics": s, "val_acc": 0.95, "val_loss": 0.2}
         for m in MODELS for e in EPOCHS for s in ("adapting", "frozen")]))

    # Not-food check: non-food photos get a weaker top logit than food photos, with some overlap.
    sources = ["flowers102", "oxford_pets"] * 50
    val_ood, test_ood = rng.normal(0, 1, (100, 3)), rng.normal(0, 1, (100, 3))
    for z in (val_ood, test_ood):
        z[np.arange(100), rng.integers(0, 3, 100)] += rng.normal(4.0, 1.2, 100)
    ood = evaluate(val_logits, val_ood, test_logits, test_ood, sources, calibration["temperature"], 0.95, 300)
    ood["sources"] = {k: {**SOURCES[k], "available": 3000} for k in ("flowers102", "oxford_pets")}
    store.write_json(Files.OOD, ood)
    store.write_csv(Files.OOD_SCORES, score_table(
        {"val_food": val_logits, "val_non_food": val_ood, "test_food": test_logits, "test_non_food": test_ood},
        {"val_food": ["foodvision"] * 90, "val_non_food": sources, "test_food": ["foodvision"] * 150,
         "test_non_food": sources}, calibration["temperature"]))
    store.write_json(Files.MODEL_META, {
        "arch": name, "classes": CLASSES, "label": f"EfficientNet-B{name[-1]}", "weights": "IMAGENET1K_V1",
        "epochs": epochs, "seed": 42, "temperature": calibration["temperature"],
        "trained_on": "360 images (stratified training split)",
        "metrics": {"test_accuracy": float(correct.mean()),
                    "test_accuracy_wilson": list(wilson_interval(int(correct.sum()), 150)),
                    "test_macro_f1": report["macro_f1"], "test_ece_uncalibrated": calibration["uncalibrated"]["ece"],
                    "test_ece_scaled": calibration["temperature_scaled"]["ece"]},
        "transforms": {"resize": 320, "crop": 300, "interpolation": "bicubic"},
        "ood": {k: ood[k] for k in ("score", "threshold", "temperature", "target_tpr")},
        "environment": {"python": "3.x", "foodvision": "synthetic"}, "config_fingerprint": "synthetic",
        "created_at": "synthetic", "file": {"path": "model/foodvision.pt", "bytes": 310_000, "sha256": "0" * 64},
    })

    gallery = store.path("gallery")
    gallery.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, (kind, ok) in enumerate((("misclassified", False), ("most confident", True), ("least confident", True))):
        for suffix in ("view", "cam"):
            Image.new("RGB", (64, 64), (40 * i, 90, 140)).save(gallery / f"{i:02d}_{suffix}.jpg")
        entries.append({"view": f"{i:02d}_view.jpg", "cam": f"{i:02d}_cam.jpg", "label": "pizza",
                        "prediction": "pizza" if ok else "steak", "confidence": 0.9, "correct": ok, "kind": kind})
    store.write_json(Files.GALLERY, {"model": name, "epochs": epochs, "entries": entries})
    store.write_json(Files.MANIFEST, {"schema_version": SCHEMA_VERSION, "synthetic": True, "stages": {},
                                      "environment": {"python": "3.x"}, "updated_at": "synthetic"})
    _analysis_artifacts(store, rng, ood)
    write_reports(store)
    summary_figure(store)
    return store


def _analysis_artifacts(store: ArtifactStore, rng: np.random.Generator, ood: dict) -> None:
    """Robustness, feature map and data-efficiency results with plausible shapes (accuracy falls with severity)."""
    drops = {"blur": 0.05, "noise": 0.06, "dark": 0.03, "jpeg": 0.02, "lowres": 0.09}
    rows = []
    for kind, severity in conditions():
        accuracy = 0.96 - (drops[kind] * severity if kind != "none" else 0.0)
        correct = int(round(accuracy * 150))
        low, high = wilson_interval(correct, 150)
        rows.append({"corruption": kind, "label": "Clean" if kind == "none" else CORRUPTIONS[kind]["label"],
                     "severity": severity, "level": None if kind == "none" else CORRUPTIONS[kind]["levels"][severity - 1],
                     "accuracy": correct / 150, "ci_low": low, "ci_high": high, "mean_confidence": 0.9 - 0.05 * severity,
                     "n": 150, "accepted_as_food": 0.95 - 0.08 * severity})
    store.write_csv(Files.ROBUSTNESS, pd.DataFrame(rows))

    centres = {"pizza": (-20, 5), "steak": (15, 18), "sushi": (10, -20), "flowers": (35, -5), "cats and dogs": (-5, 35)}
    points = []
    for label, (cx, cy) in centres.items():
        n = 50
        food = label in CLASSES
        for i in range(n):
            wrong = food and i < 2
            prediction = CLASSES[(CLASSES.index(label) + 1) % 3] if wrong else (label if food else "steak")
            points.append({"x": cx + rng.normal(0, 4), "y": cy + rng.normal(0, 4), "group": "food" if food else "non-food",
                           "label": label, "prediction": prediction, "confidence": 0.9,
                           "outcome": ("misclassified" if wrong else "correct") if food else "non-food",
                           "accepted_as_food": bool(food or i < 5)})
    store.write_csv(Files.FEATURE_MAP, pd.DataFrame(points))

    efficiency = [{"fraction": f, "n_train": n, "seed": s, "val_acc": min(1.0, acc + rng.normal(0, 0.01)), "val_loss": 0.3,
                   "source": "research runs" if f == 1.0 else "this analysis"}
                  for f, n, acc in ((0.1, 36, 0.82), (0.25, 90, 0.89), (0.5, 180, 0.93), (1.0, 360, 0.96))
                  for s in (42, 43, 44)]
    store.write_csv(Files.DATA_EFFICIENCY, pd.DataFrame(efficiency))
    store.write_json(Files.ANALYSIS, {
        "checkpoint_check": {"reloaded_clean_accuracy": 0.96, "reported_test_accuracy": 0.96, "match": True},
        "robustness": {"corruptions": {k: {"label": v["label"], "levels": list(v["levels"]), "unit": v["unit"]}
                                       for k, v in CORRUPTIONS.items()}, "note": "synthetic"},
        "feature_map": {"method": "synthetic", "perplexity": 30},
        "data_efficiency": {"model": "effnetb3", "epochs": 20, "note": "synthetic"},
    })
