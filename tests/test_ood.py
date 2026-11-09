from pathlib import Path

import numpy as np
import pytest

from foodvision.calibration import softmax
from foodvision.ood import (
    SCORES,
    auroc,
    auroc_interval,
    evaluate,
    fpr_at_tpr,
    sample_split,
    score_table,
    scores,
    single_score,
    threshold_for_tpr,
)


def _logits(n, bump, rng):
    z = rng.normal(0, 1, (n, 3))
    z[np.arange(n), rng.integers(0, 3, n)] += bump
    return z


def test_scores_match_their_definitions():
    scipy_special = pytest.importorskip("scipy.special")
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, (50, 3))
    s = scores(z, temperature=1.7)
    np.testing.assert_allclose(s["msp"], softmax(z, 1.7).max(axis=1))
    np.testing.assert_allclose(s["max_logit"], z.max(axis=1))
    np.testing.assert_allclose(s["energy"], scipy_special.logsumexp(z, axis=1))
    assert single_score(z[3], "energy", 1.7) == pytest.approx(s["energy"][3])
    with pytest.raises(ValueError):
        single_score(z[0], "not-a-score")


def test_scores_are_stable_for_huge_logits():
    s = scores(np.array([[1000.0, 0.0, -1000.0]]))
    assert np.isfinite(s["energy"]).all() and s["energy"][0] == pytest.approx(1000.0)


def test_auroc_matches_scikit_learn_including_ties():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    for x, y in [
        (rng.normal(1, 1, 120), rng.normal(0, 1, 90)),
        (rng.integers(0, 4, 60).astype(float), rng.integers(0, 4, 40).astype(float)),  # many ties
    ]:
        labels = np.r_[np.ones(len(x)), np.zeros(len(y))]
        assert auroc(x, y) == pytest.approx(metrics.roc_auc_score(labels, np.r_[x, y]))


def test_auroc_edge_cases():
    assert auroc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert auroc([0.0], [1.0]) == 0.0
    assert auroc([1.0], [1.0]) == 0.5
    assert np.isnan(auroc([], [1.0]))


def test_threshold_keeps_at_least_the_target_share():
    rng = np.random.default_rng(2)
    for n in (1, 7, 20, 90, 150):
        for target in (0.5, 0.9, 0.95, 0.99):
            x = rng.normal(0, 1, n)
            t = threshold_for_tpr(x, target)
            assert (x >= t).mean() >= target - 1e-12
            assert t in x  # an observed score, so the rule is easy to explain
    with pytest.raises(ValueError):
        threshold_for_tpr([], 0.95)


def test_fpr_at_tpr_on_a_hand_checked_example():
    food = np.arange(1, 21, dtype=float)  # 1..20; keeping 95% rejects only the lowest score: threshold 2
    non_food = np.array([0.0, 1.0, 1.5, 2.0, 5.0])
    assert threshold_for_tpr(food, 0.95) == 2.0
    assert fpr_at_tpr(food, non_food, 0.95) == pytest.approx(2 / 5)


def test_auroc_interval_contains_the_estimate():
    rng = np.random.default_rng(3)
    ci = auroc_interval(rng.normal(1, 1, 80), rng.normal(0, 1, 60), n_resamples=400, seed=0)
    assert 0.0 <= ci["low"] <= ci["estimate"] <= ci["high"] <= 1.0


def test_sample_split_is_reproducible_halved_and_disjoint():
    files = {"flowers102": [Path(f"f{i}.jpg") for i in range(50)], "oxford_pets": [Path(f"p{i}.jpg") for i in range(7)]}
    val, test = sample_split(files, per_source=20, seed=42)
    assert (val, test) == sample_split(files, per_source=20, seed=42)
    assert not {p for p, _ in val} & {p for p, _ in test}
    assert sum(s == "flowers102" for _, s in val) == 10 and sum(s == "flowers102" for _, s in test) == 10
    assert sum(s == "oxford_pets" for _, s in val) == 3 and sum(s == "oxford_pets" for _, s in test) == 4


def test_evaluate_selects_on_validation_and_reports_test_once():
    rng = np.random.default_rng(4)
    val_food, test_food = _logits(90, 6.0, rng), _logits(150, 6.0, rng)
    val_ood, test_ood = _logits(100, 3.5, rng), _logits(100, 3.5, rng)
    sources = ["flowers102"] * 60 + ["oxford_pets"] * 40
    report = evaluate(val_food, val_ood, test_food, test_ood, sources, temperature=1.3, n_resamples=200)
    assert report["status"] == "ok" and report["score"] in SCORES
    chosen = next(r for r in report["scores"] if r["score"] == report["score"])
    assert chosen["val_auroc"] == max(r["val_auroc"] for r in report["scores"])
    assert report["validation"]["accepted_food"] >= 0.95
    assert report["test"]["auroc"]["estimate"] > 0.8
    assert {r["n"] for r in report["per_source"]} == {60, 40}
    for key in ("accepted_food", "rejected_non_food"):
        assert 0.0 <= report["test"][key] <= 1.0
    table = score_table({"test_food": test_food, "test_non_food": test_ood},
                        {"test_food": ["foodvision"] * 150, "test_non_food": sources}, 1.3)
    assert set(table["group"]) == {"food", "non-food"} and set(SCORES) <= set(table.columns)
    assert len(table) == 250
