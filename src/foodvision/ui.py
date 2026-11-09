"""Streamlit helpers shared by the app pages: cached artifact access, page chrome and empty states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .artifacts import ArtifactStore

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500&display=swap');
.fv-hero h1, .fv-title {font-family: 'IBM Plex Serif', Georgia, serif; font-weight: 500; letter-spacing: -0.01em;}
.fv-hero h1 {font-size: 2.55rem; line-height: 1.15; margin: 0.2rem 0 0.8rem 0; color: #1B2430;}
.fv-lede {font-family: 'IBM Plex Sans', sans-serif; font-size: 1.1rem; line-height: 1.6; color: #3A4552; max-width: 64ch;}
.fv-finding {font-family: 'IBM Plex Sans', sans-serif; border-left: 3px solid #E3A21A; padding: 0.15rem 0 0.15rem 0.9rem;
             margin: 0 0 1rem 0; line-height: 1.55; max-width: 72ch;}
.fv-note {color: #5B6573; font-size: 0.92rem; line-height: 1.5; max-width: 80ch;}
</style>
"""


def store() -> ArtifactStore:
    return ArtifactStore()


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _json(path: str, mtime: float) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _text(path: str, mtime: float) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_text(name: str) -> str:
    path = store().path(name)
    return _text(str(path), _mtime(path))


def read_json(name: str) -> Any:
    path = store().path(name)
    return _json(str(path), _mtime(path))


def read_csv(name: str) -> pd.DataFrame:
    path = store().path(name)
    return _csv(str(path), _mtime(path))


def available(*names: str) -> bool:
    return not store().missing(*names)


def require(*names: str) -> None:
    """Stop the page with directions when the pipeline has not produced the needed files yet."""
    missing = store().missing(*names)
    if missing:
        st.info(
            "These results have not been generated yet. Run the pipeline once on a GPU "
            "(`notebooks/run_pipeline_colab.ipynb`), then commit the `artifacts/` folder. "
            f"Missing: {', '.join(f'`{m}`' for m in missing)}."
        )
        st.stop()


def missing_notice(*names: str) -> bool:
    """Show directions inside a tab or section and return True when results are missing (does not stop the page)."""
    missing = store().missing(*names)
    if missing:
        st.info(f"Not generated yet ({', '.join(f'`{m}`' for m in missing)}). Run the pipeline and commit `artifacts/`.")
    return bool(missing)


def page(title: str, intro: str) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"<h1 class='fv-title'>{title}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='fv-lede'>{intro}</p>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<p class='fv-note'>{text}</p>", unsafe_allow_html=True)


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{100 * value:.{digits}f}%"
