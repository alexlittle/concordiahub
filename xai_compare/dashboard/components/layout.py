
import streamlit as st
from .sidebar import sidebar
from .styles import load_css

def apply_layout(title: str):
    # Page title
    st.set_page_config(page_title=title, layout="wide")

    # inject css
    load_css()

    # Sidebar
    data = sidebar()

    # Page title
    st.title(title)

    return data


