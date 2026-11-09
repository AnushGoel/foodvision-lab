import pytest

go = pytest.importorskip("plotly.graph_objects")

from foodvision import viz  # noqa: E402
from foodvision.artifacts import ArtifactStore, Files  # noqa: E402
from foodvision.calibration import abstention_curve  # noqa: E402


def test_every_figure_builds_and_validates(artifacts_dir):
    store = ArtifactStore(artifacts_dir)
    summary = store.read_csv(Files.ASSIGNMENT_SUMMARY)
    report = store.read_json(Files.TEST_REPORT)
    calibration = store.read_json(Files.CALIBRATION)
    predictions = store.read_csv(Files.PREDICTIONS)
    audit = store.read_json(Files.AUDIT)
    probs = predictions[[c for c in predictions.columns if c.startswith("p_")]].to_numpy()
    labels = predictions["label"].map({"pizza": 0, "steak": 1, "sushi": 2}).to_numpy()
    figures = [
        viz.scenario_heatmap(summary, "test_loss", "Test loss", "{:.3f}"),
        viz.interval_plot(summary, "test_acc", "test_ci_low", "test_ci_high", "EfficientNet-B3, 30 ep"),
        viz.pareto_plot(summary, "train_seconds", "test_acc", "Training time (s)"),
        viz.learning_curves(store.read_json(Files.ASSIGNMENT_HISTORY), ["effnetb2_15"], "acc", ["train", "test"]),
        viz.confusion_heatmap(report["confusion_matrix"], report["classes"], normalized=True),
        viz.per_class_bars(report["per_class"]),
        viz.reliability_diagram(calibration["uncalibrated"]["bins"], calibration["temperature_scaled"]["bins"]),
        viz.confidence_histogram(predictions),
        viz.abstention_plot(abstention_curve(probs, labels), 0.7),
        viz.probability_bars(["pizza", "steak", "sushi"], [0.7, 0.2, 0.1]),
        viz.latency_bars(store.read_json(Files.BENCHMARK)),
        viz.class_counts(audit["counts"]),
        viz.size_scatter(store.read_csv(Files.IMAGES)),
    ]
    for figure in figures:
        assert isinstance(figure, go.Figure)
        assert figure.to_dict()["data"]


def test_ood_histogram_overlays_both_groups_with_shared_bins(artifacts_dir):
    store = ArtifactStore(artifacts_dir)
    ood = store.read_json(Files.OOD)
    scores = store.read_csv(Files.OOD_SCORES)
    fig = viz.ood_histogram(scores[scores["split"] == "test"], ood["score"], ood["threshold"], ood["score_label"])
    assert len(fig.data) == 2 and fig.layout.barmode == "overlay"
    assert fig.data[0].xbins.size == fig.data[1].xbins.size


def test_analysis_figures_build(artifacts_dir):
    store = ArtifactStore(artifacts_dir)
    robustness = store.read_csv(Files.ROBUSTNESS)
    assert len(viz.robustness_plot(robustness).data) == 5
    assert len(viz.robustness_plot(robustness, "accepted_as_food").data) == 5
    feature_map = viz.feature_map_plot(store.read_csv(Files.FEATURE_MAP))
    assert {trace.name for trace in feature_map.data} >= {"pizza", "steak", "sushi", "misclassified"}
    assert len(viz.data_efficiency_plot(store.read_csv(Files.DATA_EFFICIENCY)).data) == 2
