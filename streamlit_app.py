"""FoodVision Lab: Streamlit entry point (Streamlit Community Cloud runs this file)."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="FoodVision Lab", page_icon="🍣", layout="wide")

PAGES = ROOT / "app_pages"
navigation = st.navigation(
    {
        "Results": [
            st.Page(PAGES / "overview.py", title="Overview", icon=":material/home:", default=True),
            st.Page(PAGES / "experiments.py", title="Experiment lab", icon=":material/science:"),
            st.Page(PAGES / "training.py", title="Training curves", icon=":material/show_chart:"),
        ],
        "The selected model": [
            st.Page(PAGES / "diagnostics.py", title="Errors and diagnostics", icon=":material/grid_on:"),
            st.Page(PAGES / "reliability.py", title="Reliability", icon=":material/verified:"),
            st.Page(PAGES / "predict.py", title="Try the model", icon=":material/photo_camera:"),
        ],
        "About": [st.Page(PAGES / "methods.py", title="Data and methods", icon=":material/menu_book:")],
    }
)
navigation.run()
