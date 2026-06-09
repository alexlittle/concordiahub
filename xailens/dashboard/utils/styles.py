# xailens/dashboard/components/styles.py

from pathlib import Path
import streamlit as st

def load_css():
    css_path = Path(__file__).parent.parent / "assets" / "styles.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
