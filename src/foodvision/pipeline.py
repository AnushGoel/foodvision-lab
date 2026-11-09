"""Command-line pipeline: audit -> assignment protocol -> research protocol -> benchmark -> artifacts.

    python -m foodvision.pipeline --config configs/default.yaml        # all stages (GPU recommended)
    python -m foodvision.pipeline --stages research --force            # re-run one stage

A stage is skipped only when its outputs exist *and* were produced under the same configuration fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any

import torch
import torchvision

from . import __version__
from .artifacts import SCHEMA_VERSION, ArtifactStore, Files
from .audit import audit_dataset
from .config import Config, load_config
from .analysis import CORRUPTIONS
from .experiments import (
    DataBundle,
    benchmark_latency,
    data_efficiency,
    feature_map,
    make_gallery,
    prepare_data,
    robustness_eval,
    run_assignment,
    run_research,
)
from .figures import summary_figure
from .models import REGISTRY, architecture_summary, get_transforms, load_checkpoint, save_checkpoint
from .reports import write_reports

LOG = logging.getLogger("foodvision")
STAGES = ("audit", "assignment", "research", "analysis", "benchmark")
SECTIONS = {
    "audit": ("data",),
    "assignment": ("seed", "models", "epochs", "data", "train"),
    "research": ("seed", "models", "epochs", "data", "train", "research", "evaluation", "ood"),
    "analysis": ("seed", "models", "epochs", "data", "train", "research", "evaluation", "ood", "analysis"),
    "benchmark": ("models", "evaluation"),
}
DEPENDS = {"analysis": "research"}  # re-run a stage when its input stage ran in the same invocation
OUTPUTS = {
    "audit": (Files.AUDIT, Files.IMAGES, Files.ARCHITECTURE),
    "assignment": (Files.ASSIGNMENT_SUMMARY, Files.ASSIGNMENT_HISTORY),
    "research": (Files.VALIDATION, Files.SELECTION, Files.TEST_REPORT, Files.PREDICTIONS, Files.CALIBRATION,
                 Files.OOD, Files.MODEL, Files.MODEL_META),
    "analysis": (Files.ROBUSTNESS, Files.FEATURE_MAP, Files.DATA_EFFICIENCY, Files.ANALYSIS),
    "benchmark": (Files.BENCHMARK,),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def environment() -> dict[str, Any]:
    """Software and hardware details recorded with every artifact set."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),  # TorchVersion is a str subclass; keep metadata plain
        "torchvision": str(torchvision.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
        "foodvision": __version__,
    }


def _audit(cfg: Config, bundle: DataBundle, store: ArtifactStore) -> None:
    summary, frame = audit_dataset(bundle.root)
    store.write_json(Files.AUDIT, summary)
    store.write_csv(Files.IMAGES, frame)
    store.write_json(Files.ARCHITECTURE, [architecture_summary(name) for name in cfg.models])


def _assignment(cfg: Config, bundle: DataBundle, store: ArtifactStore, device: torch.device) -> None:
    frame, histories = run_assignment(cfg, bundle, device)
    store.write_csv(Files.ASSIGNMENT_SUMMARY, frame)
    store.write_json(Files.ASSIGNMENT_HISTORY, histories)


def _research(cfg: Config, bundle: DataBundle, store: ArtifactStore, device: torch.device) -> None:
    result = run_research(cfg, bundle, device)
    store.write_csv(Files.VALIDATION, result.validation)
    store.write_json(Files.RESEARCH_HISTORY, result.histories)
    store.write_json(Files.SELECTION, result.selection)
    store.write_json(Files.TEST_REPORT, result.test_report)
    store.write_csv(Files.PREDICTIONS, result.predictions)
    store.write_json(Files.CALIBRATION, result.calibration)
    if result.bn_ablation is not None:
        store.write_csv(Files.BN_ABLATION, result.bn_ablation)
    store.write_json(Files.OOD, result.ood)
    if result.ood_scores is not None:
        store.write_csv(Files.OOD_SCORES, result.ood_scores)
    else:
        store.path(Files.OOD_SCORES).unlink(missing_ok=True)  # no stale scores from an earlier run
    if cfg.evaluation.export_gallery:
        shutil.rmtree(store.path("gallery"), ignore_errors=True)  # no stale images from earlier runs
        entries = make_gallery(result, store.path("gallery"), cfg.evaluation.gallery_size)
        store.write_json(Files.GALLERY, {"model": result.model_name, "epochs": result.epochs, "entries": entries})
    tf = get_transforms(result.model_name)
    report = result.test_report
    meta = {
        "label": REGISTRY[result.model_name].label,
        "weights": REGISTRY[result.model_name].weights.name,
        "epochs": result.epochs,
        "seed": cfg.seed,
        "temperature": result.calibration["temperature"],
        "trained_on": f"{result.selection['split']['train']} images (stratified training split)",
        "metrics": {
            "test_accuracy": report["accuracy"],
            "test_accuracy_wilson": report["accuracy_wilson"],
            "test_macro_f1": report["macro_f1"],
            "test_ece_uncalibrated": result.calibration["uncalibrated"]["ece"],
            "test_ece_scaled": result.calibration["temperature_scaled"]["ece"],
        },
        "transforms": {"resize": tf.resize_size[0], "crop": tf.crop_size[0], "interpolation": str(tf.interpolation.value)},
        # the not-food check travels with the model, so the app applies exactly the validated threshold
        "ood": {key: result.ood[key] for key in ("score", "threshold", "temperature", "target_tpr")}
        if result.ood.get("status") == "ok" else None,
        "environment": environment(),
        "config_fingerprint": cfg.fingerprint(*SECTIONS["research"]),
        "created_at": _now(),
    }
    info = save_checkpoint(result.model, store.path(Files.MODEL), result.model_name, report["classes"], meta)
    store.write_json(Files.MODEL_META, {"arch": result.model_name, "classes": report["classes"], **meta, "file": info})


def _analysis(cfg: Config, bundle: DataBundle, store: ArtifactStore, device: torch.device) -> None:
    missing = store.missing(Files.MODEL, Files.SELECTION, Files.VALIDATION, Files.TEST_REPORT)
    if missing:
        raise RuntimeError(f"Run the research stage first; missing {missing}")
    # Reload the saved model exactly the way the app does, which also verifies the checkpoint.
    model, payload = load_checkpoint(store.path(Files.MODEL), device=device)
    name, meta = payload["arch"], payload["meta"]
    temperature = float(meta.get("temperature", 1.0))
    robustness = robustness_eval(model, name, bundle.test, temperature, meta.get("ood"), device, cfg.seed,
                                 cfg.train.batch_size)
    store.write_csv(Files.ROBUSTNESS, robustness)
    clean = float(robustness.loc[robustness["corruption"] == "none", "accuracy"].iloc[0])
    reported = float(store.read_json(Files.TEST_REPORT)["accuracy"])
    match = abs(clean - reported) <= 1.0 / len(bundle.test) + 1e-9  # allows one image of GPU non-determinism
    if match:
        LOG.info("Checkpoint check passed: the reloaded model scores %.1f%% (reported %.1f%%)", 100 * clean, 100 * reported)
    else:
        LOG.warning("Checkpoint check FAILED: the reloaded model scores %.1f%%, reported %.1f%%", 100 * clean,
                    100 * reported)
    store.write_csv(Files.FEATURE_MAP, feature_map(cfg, model, name, bundle, temperature, meta.get("ood"), device))
    selection = store.read_json(Files.SELECTION)
    efficiency = data_efficiency(cfg, bundle, selection, store.read_csv(Files.VALIDATION), device)
    store.write_csv(Files.DATA_EFFICIENCY, efficiency)
    store.write_json(
        Files.ANALYSIS,
        {
            "checkpoint_check": {"reloaded_clean_accuracy": clean, "reported_test_accuracy": reported, "match": match},
            "robustness": {
                "corruptions": {k: {"label": v["label"], "levels": list(v["levels"]), "unit": v["unit"]}
                                for k, v in CORRUPTIONS.items()},
                "note": "Exploratory: corrupted copies of the test photos at the model's input size; no decision was "
                "made from these results.",
            },
            "feature_map": {"method": "L2-normalised pooled features, PCA to at most 50 dimensions, then t-SNE",
                            "perplexity": cfg.analysis.perplexity},
            "data_efficiency": {
                "model": name,
                "epochs": int(selection["epochs"]),
                "note": "Nested class-stratified subsets of the training split, scored on the validation split, with "
                "the same epochs and learning rate for every fraction; the 100% points are the research runs.",
            },
        },
    )


def _write_manifest(store: ArtifactStore, manifest: dict[str, Any], cfg: Config) -> None:
    files = []
    for path in sorted(p for p in store.root.rglob("*") if p.is_file() and p.name != Files.MANIFEST):
        files.append({"file": str(path.relative_to(store.root)), "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16]})
    manifest.update(
        schema_version=SCHEMA_VERSION, updated_at=_now(), environment=environment(), config=cfg.to_dict(), files=files
    )
    store.write_json(Files.MANIFEST, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the FoodVision Lab experimental pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {', '.join(STAGES)}")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    parser.add_argument("--force", action="store_true", help="re-run stages even when cached outputs match")
    args = parser.parse_args(argv)
    # force=True: notebooks often pre-configure logging, which would otherwise hide all progress output
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout, force=True)

    cfg = load_config(args.config)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if unknown := sorted(set(stages) - set(STAGES)):
        parser.error(f"unknown stage(s): {unknown}")
    store = ArtifactStore(cfg.paths.artifacts)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOG.info("FoodVision Lab %s | device %s | stages %s", __version__, device, ", ".join(stages))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    manifest = store.manifest() or {"schema_version": SCHEMA_VERSION, "stages": {}}
    bundle = prepare_data(cfg)

    ran: set[str] = set()
    for stage in stages:
        fingerprint = cfg.fingerprint(*SECTIONS[stage])
        cached = manifest["stages"].get(stage, {}).get("fingerprint") == fingerprint
        stale = DEPENDS.get(stage) in ran
        if cached and not stale and not args.force and not store.missing(*OUTPUTS[stage]):
            LOG.info("Stage %s is up to date (config %s); use --force to re-run it", stage, fingerprint)
            continue
        start = time.perf_counter()
        if stage == "audit":
            _audit(cfg, bundle, store)
        elif stage == "assignment":
            _assignment(cfg, bundle, store, device)
        elif stage == "research":
            _research(cfg, bundle, store, device)
        elif stage == "analysis":
            _analysis(cfg, bundle, store, device)
        else:
            store.write_json(Files.BENCHMARK, benchmark_latency(cfg.models, cfg.evaluation.latency_runs))
        seconds = time.perf_counter() - start
        ran.add(stage)
        manifest["stages"][stage] = {"fingerprint": fingerprint, "completed_at": _now(), "seconds": round(seconds, 1)}
        _write_manifest(store, manifest, cfg)
        LOG.info("Stage %s finished in %.0f s", stage, seconds)

    write_reports(store)
    if (figure := summary_figure(store)) is not None:
        LOG.info("Summary figure written to %s", figure)
    _write_manifest(store, manifest, cfg)
    LOG.info("Artifacts written to %s", store.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
