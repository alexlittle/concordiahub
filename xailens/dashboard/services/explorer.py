
import streamlit as st

from xailens.run_explorer import RunExplorer

@st.cache_resource
def get_explorer():
    return RunExplorer()
