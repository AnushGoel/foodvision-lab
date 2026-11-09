"""Shared fixtures: synthetic image folders and a complete synthetic artifact set."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic import make_artifacts, make_image_folder  # noqa: E402


@pytest.fixture
def image_root(tmp_path: Path) -> Path:
    return make_image_folder(tmp_path / "data")


@pytest.fixture
def artifacts_dir(tmp_path: Path, image_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "artifacts"
    make_artifacts(root, image_root)
    monkeypatch.setenv("FOODVISION_ARTIFACTS", str(root))
    return root
