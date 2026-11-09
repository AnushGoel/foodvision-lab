from pathlib import Path

import pytest

from foodvision.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_matches_the_assignment():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.models == ("effnetb2", "effnetb3", "effnetb4")
    assert cfg.epochs == (15, 20, 30)
    assert len(cfg.data.urls) >= 2


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  learnig_rate: 0.01\n")
    with pytest.raises(ValueError, match="learnig_rate"):
        load_config(path)


def test_fingerprint_changes_only_with_relevant_sections():
    base = Config()
    assert base.fingerprint("train") == Config().fingerprint("train")
    changed = Config(seed=7)
    assert changed.fingerprint("train") == base.fingerprint("train")
    assert changed.fingerprint("seed", "train") != base.fingerprint("seed", "train")


def test_main_seed_must_be_a_research_seed(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text("seed: 7\nresearch:\n  seeds: [1, 2]\n")
    with pytest.raises(ValueError, match="research.seeds"):
        load_config(path)


def test_smoke_config_is_a_valid_tiny_run():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    assert cfg.models == ("effnetb2",) and cfg.research.seeds == (42,)
    assert cfg.paths.artifacts != load_config(ROOT / "configs" / "default.yaml").paths.artifacts


@pytest.mark.parametrize(
    "block, message",
    [
        ("ood:\n  sources: [imagenet]\n", "ood.sources"),
        ("ood:\n  sources: []\n", "ood.sources"),
        ("ood:\n  per_source: 2\n", "ood.per_source"),
        ("ood:\n  target_tpr: 1.0\n", "ood.target_tpr"),
        ("ood:\n  treshold: 0.5\n", "treshold"),
    ],
)
def test_ood_settings_are_validated(tmp_path, block, message):
    path = tmp_path / "ood.yaml"
    path.write_text(block)
    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.parametrize("block", ["analysis:\n  fractions: []\n", "analysis:\n  fractions: [0.0, 1.0]\n",
                                   "analysis:\n  fractions: [1.5]\n", "analysis:\n  perplexity: 1\n"])
def test_analysis_settings_are_validated(tmp_path, block):
    path = tmp_path / "analysis.yaml"
    path.write_text(block)
    with pytest.raises(ValueError, match="analysis"):
        load_config(path)


def test_ood_settings_are_ignored_when_the_check_is_off(tmp_path):
    path = tmp_path / "off.yaml"
    path.write_text("ood:\n  enabled: false\n  per_source: 0\n")
    assert load_config(path).ood.enabled is False
