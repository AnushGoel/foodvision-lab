import numpy as np

from foodvision.artifacts import SCHEMA_VERSION, ArtifactStore, Files
from foodvision.reports import audit_markdown, experiments_markdown, model_card_markdown, write_reports


def test_json_is_sanitised_and_atomic(tmp_path):
    store = ArtifactStore(tmp_path)
    store.write_json("x.json", {"a": np.float32(1.5), "b": [np.nan, 2], "c": np.arange(2)})
    assert store.read_json("x.json") == {"a": 1.5, "b": [None, 2], "c": [0, 1]}
    assert not list(tmp_path.glob("*.tmp"))


def test_manifest_schema_check(tmp_path):
    store = ArtifactStore(tmp_path)
    assert store.manifest() is None
    store.write_json(Files.MANIFEST, {"schema_version": SCHEMA_VERSION + 1})
    assert store.manifest() is None


def test_reports_render_from_synthetic_artifacts(artifacts_dir):
    store = ArtifactStore(artifacts_dir)
    write_reports(store)
    experiments = experiments_markdown(store)
    assert "Assignment protocol" in experiments and "Research protocol" in experiments and "McNemar" in experiments
    assert "Images per class" in audit_markdown(store)
    card = model_card_markdown(store)
    for section in ("## Model details", "## Intended use", "**Out of scope:**", "## Factors", "## Metrics",
                    "## Evaluation data", "## Training data", "## Quantitative analyses", "## Ethical considerations",
                    "## Caveats and recommendations"):
        assert section in card, section
    assert "Not-food check" in card and "allergies" in card and "Robustness" in card
    for section in ("Analysis of the selected model", "Robustness to everyday photo problems", "Data efficiency",
                    "Checkpoint check", "Feature map"):
        assert section in experiments, section
    assert store.exists("reports/EXPERIMENTS.md")


def test_to_builtin_returns_exact_builtin_types():
    """Regression: torch.__version__ is a str subclass that torch.load(weights_only=True) refuses to unpickle."""
    import pickle
    from pathlib import Path

    from foodvision.artifacts import to_builtin

    class VersionString(str):
        pass

    clean = to_builtin({"torch": VersionString("2.6.0"), "t": np.float32(1.5), "ok": np.bool_(True),
                        "path": Path("a"), "items": (VersionString("x"), np.int64(2))})
    assert clean == {"torch": "2.6.0", "t": 1.5, "ok": True, "path": "a", "items": ["x", 2]}
    assert type(clean["torch"]) is str and type(clean["items"][0]) is str and clean["ok"] is True
    assert b"VersionString" not in pickle.dumps(clean)
