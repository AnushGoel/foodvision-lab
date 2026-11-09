import numpy as np
import pytest

from foodvision.metrics import classification_report, confusion_matrix, macro_f1, normalize_confusion


def test_confusion_matrix_rows_are_true_labels():
    cm = confusion_matrix([0, 0, 1, 2, 2], [0, 1, 1, 2, 0], 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]
    assert normalize_confusion(cm).sum(axis=1) == pytest.approx([1, 1, 1])


def test_report_values_for_the_assignment_best_model():
    cm = [[45, 1, 0], [0, 57, 1], [0, 1, 45]]  # EfficientNet-B3, 15 epochs, from the assignment run
    y_true = [i for i, row in enumerate(cm) for _ in range(sum(row))]
    y_pred = [j for row in cm for j, n in enumerate(row) for _ in range(n)]
    report = classification_report(y_true, y_pred, ["pizza", "steak", "sushi"])
    assert report["accuracy"] == pytest.approx(147 / 150)
    precision = [c["precision"] for c in report["per_class"]]
    assert precision == pytest.approx([1.0, 57 / 59, 45 / 46])
    assert report["macro_f1"] == pytest.approx(0.98054, abs=1e-4)


def test_empty_class_gives_zero_not_nan():
    report = classification_report([0, 0], [0, 0], ["a", "b"])
    assert report["per_class"][1] == {"class": "b", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    assert not np.isnan(report["macro_f1"])


def test_macro_f1_helper_matches_report():
    y_true, y_pred = [0, 1, 2, 2, 1], [0, 2, 2, 2, 1]
    assert macro_f1(y_true, y_pred, 3) == pytest.approx(classification_report(y_true, y_pred, ["a", "b", "c"])["macro_f1"])
