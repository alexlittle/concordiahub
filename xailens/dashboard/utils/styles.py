"""
Functions for applying CSS to Streamlit
"""

from pathlib import Path
import streamlit as st

def load_css():
    """
    Injects CSS file into the app
    """
    css_path = Path(__file__).parent.parent / "assets" / "styles.css"
    with open(css_path) as f:
        st.html(f"<style>{f.read()}</style>")

