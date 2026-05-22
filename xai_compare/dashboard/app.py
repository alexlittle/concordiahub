

# turn off transformer verbose messages
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


import streamlit as st

from xai_compare.run_explorer import RunExplorer
from xai_compare.dashboard.components.layout import apply_layout

@st.cache_resource
def get_explorer():
    return RunExplorer()

explorer = get_explorer()

apply_layout("XAI Compare")
st.write("Welcome to XAI Compare")

recent_runs = explorer.get_comparable_runs()

for rr in recent_runs:
    st.write(rr)

