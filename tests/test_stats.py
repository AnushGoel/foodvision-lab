import numpy as np
import pytest

from foodvision.stats import bootstrap_ci, holm_correction, mcnemar_exact, paired_bootstrap_difference, wilson_interval


def test_wilson_matches_reference_values():
    low, high = wilson_interval(147, 150)
    assert low == pytest.approx(0.94285, abs=1e-4)
    assert high == pytest.approx(0.99317, abs=1e-4)


def test_wilson_stays_inside_unit_interval():
    assert wilson_interval(0, 20)[0] == 0.0
    assert wilson_interval(20, 20)[1] == pytest.approx(1.0)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_mcnemar_known_value():
    # b = 5 images only A got right, c = 1 only B got right: two-sided exact p = 2 * P(X <= 1 | n=6) = 0.21875
    a = np.array([1] * 5 + [0] * 1 + [1] * 10, dtype=bool)
    b = np.array([0] * 5 + [1] * 1 + [1] * 10, dtype=bool)
    result = mcnemar_exact(a, b)
    assert (result["a_only_correct"], result["b_only_correct"]) == (5, 1)
    assert result["p_value"] == pytest.approx(0.21875)


def test_mcnemar_no_disagreement_and_shape_check():
    same = np.ones(10, dtype=bool)
    assert mcnemar_exact(same, same)["p_value"] == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact(same, same[:5])


def test_holm_textbook_example():
    assert holm_correction([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_bootstrap_is_reproducible_and_brackets_estimate():
    data = np.r_[np.ones(140), np.zeros(10)]
    first = bootstrap_ci(np.mean, [data], 2000, seed=3)
    assert first == bootstrap_ci(np.mean, [data], 2000, seed=3)
    assert first["low"] <= first["estimate"] <= first["high"]


def test_paired_difference_of_identical_models_is_zero():
    correct = np.r_[np.ones(90, dtype=bool), np.zeros(10, dtype=bool)]
    result = paired_bootstrap_difference(correct, correct, 500)
    assert result["estimate"] == result["low"] == result["high"] == 0.0
