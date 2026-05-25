
import streamlit as st

from xai_compare.run_explorer import RunExplorer

@st.cache_resource
def get_explorer():
    return RunExplorer()
