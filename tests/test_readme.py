"""The README's results table is generated from artifacts/, so it must match the committed results."""

from pathlib import Path

import pytest

from foodvision.artifacts import ArtifactStore, Files
from foodvision.reports import README_END, README_START, readme_results, update_readme

ROOT = Path(__file__).resolve().parents[1]


def test_readme_results_match_the_committed_artifacts():
    store = ArtifactStore(ROOT / "artifacts")
    if not store.exists(Files.TEST_REPORT):
        pytest.skip("no pipeline results committed yet")
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert README_START in text and README_END in text, "README.md lost its results markers"
    block = text.split(README_START, 1)[1].split(README_END, 1)[0]
    assert block.strip() == readme_results(store).strip(), (
        "README results are out of date: run `python -m foodvision.reports` and commit README.md"
    )


def test_update_readme_rewrites_only_the_generated_block(tmp_path, artifacts_dir):
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\nIntro.\n\n{README_START}\nSTALE-BLOCK\n{README_END}\n\nRest.\n", encoding="utf-8")
    store = ArtifactStore(artifacts_dir)
    assert update_readme(readme, store) is True
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("# Title\n\nIntro.\n\n") and text.endswith("\n\nRest.\n")
    assert readme_results(store) in text and "STALE-BLOCK" not in text
    assert update_readme(readme, store) is False  # running it again changes nothing


def test_update_readme_refuses_a_readme_without_markers(tmp_path, artifacts_dir):
    readme = tmp_path / "README.md"
    readme.write_text("# No markers here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="results"):
        update_readme(readme, ArtifactStore(artifacts_dir))
