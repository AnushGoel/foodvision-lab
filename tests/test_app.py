from pathlib import Path

import pytest

pytest.importorskip("plotly")
testing = pytest.importorskip("streamlit.testing.v1")

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted((ROOT / "app_pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders_with_results(page, artifacts_dir):
    app = testing.AppTest.from_file(str(page), default_timeout=90).run()
    assert not app.exception, app.exception


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_explains_what_to_do_before_the_pipeline_runs(page, tmp_path, monkeypatch):
    monkeypatch.setenv("FOODVISION_ARTIFACTS", str(tmp_path / "nothing-here"))
    app = testing.AppTest.from_file(str(page), default_timeout=90).run()
    assert not app.exception, app.exception
    assert app.info or app.markdown


def test_entrypoint_registers_every_page():
    entry = (ROOT / "streamlit_app.py").read_text()
    for page in PAGES:
        assert f'"{page.name}"' in entry
