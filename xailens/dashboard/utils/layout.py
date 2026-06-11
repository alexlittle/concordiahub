
import streamlit as st
from xailens.dashboard.utils.sidebar import sidebar
from xailens.dashboard.utils.styles import load_css
from xailens import strings

def apply_layout(title: str):
    # Page title
    st.set_page_config(page_title=f"{strings.APP_NAME} - {title}", layout="wide")

    # inject css
    load_css()

    # Sidebar
    data = sidebar()

    # Page title
    st.html(f"<h1 class='header'>{title}</h1>")

    return data


