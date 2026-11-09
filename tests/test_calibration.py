import numpy as np
import pytest

from foodvision.calibration import (
    abstention_curve,
    brier_score,
    calibration_summary,
    expected_calibration_error,
    fit_temperature,
    nll,
    softmax,
)


def _calibrated_sample(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.5, (n, 3))
    labels = np.array([rng.choice(3, p=p) for p in softmax(logits)])
    return logits, labels


def test_softmax_rows_sum_to_one():
    assert softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])).sum(axis=1) == pytest.approx([1, 1])


def test_temperature_recovers_known_overconfidence():
    logits, labels = _calibrated_sample()
    assert fit_temperature(logits * 3.0, labels) == pytest.approx(3.0, abs=0.25)


def test_scaling_lowers_nll_and_ece_of_overconfident_model():
    logits, labels = _calibrated_sample()
    over = logits * 3.0
    t = fit_temperature(over, labels)
    assert nll(over, labels, t) < nll(over, labels)
    assert expected_calibration_error(softmax(over, t), labels) < expected_calibration_error(softmax(over), labels)


def test_ece_is_zero_when_confidence_equals_accuracy():
    probs = np.tile([0.8, 0.1, 0.1], (10, 1))
    labels = np.array([0] * 8 + [1] * 2)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_brier_perfect_prediction_is_zero():
    assert brier_score(np.eye(3), np.array([0, 1, 2])) == pytest.approx(0.0)


def test_abstention_coverage_decreases_with_threshold():
    logits, labels = _calibrated_sample(500)
    coverage = [row["coverage"] for row in abstention_curve(softmax(logits), labels)]
    assert all(a >= b for a, b in zip(coverage, coverage[1:]))


def test_summary_keeps_predictions_unchanged():
    logits, labels = _calibrated_sample(600)
    summary = calibration_summary(logits[:300], labels[:300], logits[300:], labels[300:])
    assert np.array_equal(softmax(logits[300:]).argmax(1), softmax(logits[300:], summary["temperature"]).argmax(1))
    assert {"ece", "nll", "brier", "bins"} <= set(summary["temperature_scaled"])
