from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from foodvision.analysis import (
    CORRUPTIONS,
    SEVERITIES,
    conditions,
    corrupt,
    efficiency_summary,
    nested_subset,
    project_2d,
)
from foodvision.artifacts import ArtifactStore, Files
from foodvision.figures import summary_figure


def _photo(size=(120, 90)):
    image = Image.new("RGB", size, (200, 120, 60))
    draw = ImageDraw.Draw(image)
    for i in range(0, size[0], 12):
        draw.line([(i, 0), (i, size[1])], fill=(20, 40, 160), width=3)  # sharp edges, so blur is measurable
    return image


def _difference(a, b):
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).mean())


def test_conditions_cover_every_corruption_and_severity_once():
    assert conditions()[0] == ("none", 0)
    assert len(conditions()) == 1 + len(CORRUPTIONS) * len(SEVERITIES)


@pytest.mark.parametrize("kind", sorted(CORRUPTIONS))
def test_corruptions_keep_size_and_grow_stronger(kind):
    photo = _photo()
    previous = 0.0
    for severity in SEVERITIES:
        out = corrupt(photo, kind, severity, np.random.default_rng(0))
        assert out.size == photo.size and out.mode == "RGB"
        change = _difference(photo, out)
        assert change > previous, (kind, severity)
        previous = change


def test_clean_condition_is_an_unchanged_copy_and_noise_is_reproducible():
    photo = _photo()
    clean = corrupt(photo, "none", 0, np.random.default_rng(0))
    assert clean is not photo and _difference(photo, clean) == 0.0
    a = corrupt(photo, "noise", 2, np.random.default_rng(7))
    b = corrupt(photo, "noise", 2, np.random.default_rng(7))
    assert _difference(a, b) == 0.0
    with pytest.raises(ValueError):
        corrupt(photo, "fog", 1, np.random.default_rng(0))


def test_nested_subsets_are_stratified_reproducible_and_nested():
    labels = np.repeat([0, 1, 2], [120, 120, 120])
    small, large = nested_subset(labels, 0.1, 42), nested_subset(labels, 0.5, 42)
    assert np.bincount(labels[small]).tolist() == [12, 12, 12]
    assert set(small) <= set(large)
    assert np.array_equal(small, nested_subset(labels, 0.1, 42))
    assert not np.array_equal(small, nested_subset(labels, 0.1, 43))
    assert len(nested_subset(labels, 1.0, 0)) == 360
    with pytest.raises(ValueError):
        nested_subset(labels, 0.0, 0)


def test_feature_map_projection_separates_clusters():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 1, (3, 64)) * 5
    features = np.concatenate([c + rng.normal(0, 0.3, (30, 64)) for c in centres])
    xy = project_2d(features, seed=0)
    assert xy.shape == (90, 2) and np.isfinite(xy).all()
    means = np.array([xy[i * 30:(i + 1) * 30].mean(axis=0) for i in range(3)])
    spread = max(xy[i * 30:(i + 1) * 30].std(axis=0).max() for i in range(3))
    assert min(np.linalg.norm(means[i] - means[j]) for i in range(3) for j in range(i + 1, 3)) > 2 * spread
    with pytest.raises(ValueError):
        project_2d(features[:4])


def test_efficiency_summary_averages_seeds():
    import pandas as pd

    frame = pd.DataFrame({"fraction": [0.1, 0.1, 1.0], "n_train": [36, 36, 360], "seed": [1, 2, 1],
                          "val_acc": [0.8, 0.9, 0.95]})
    summary = efficiency_summary(frame)
    assert summary["val_acc_mean"].tolist() == pytest.approx([0.85, 0.95]) and summary["runs"].tolist() == [2, 1]


def test_summary_figure_renders_from_artifacts(artifacts_dir):
    store = ArtifactStore(artifacts_dir)
    path = summary_figure(store)
    assert path == store.path(Files.SUMMARY_FIGURE) and path.stat().st_size > 50_000
    with Image.open(path) as image:
        assert image.size == (2000, 1125)  # 16:9, ready for a README or a social post


def test_summary_figure_waits_for_research_results(tmp_path):
    assert summary_figure(ArtifactStore(tmp_path)) is None
    assert not (Path(tmp_path) / Files.SUMMARY_FIGURE).exists()
