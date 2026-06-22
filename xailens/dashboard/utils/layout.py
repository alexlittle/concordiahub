
import streamlit as st
from xailens.dashboard.utils.styles import load_css
from xailens import strings
from xailens.dashboard.utils.render import sidebar




def apply_layout(title: str):
    # Page title
    st.set_page_config(page_title=f"{strings.APP_NAME} - {title}", layout="wide")

    # inject css
    load_css()

    # Sidebar
    with st.sidebar:
        st.html(sidebar.render())

    # Page title
    st.html(f"<h1 class='header'>{title}</h1>")



