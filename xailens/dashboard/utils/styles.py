# xailens/dashboard/components/styles.py

from pathlib import Path
import streamlit as st

def load_css():
    css_path = Path(__file__).parent.parent / "assets" / "styles.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def get_badge_class(model_type: str) -> str:
    t = (model_type or "").upper()
    if t == "LLM":
        return "badge badge-llm"
    if t in ("XGBOOST", "XGB"):
        return "badge badge-xgb"
    if t == "LR":
        return "badge badge-lr"
    return "badge badge-other"
