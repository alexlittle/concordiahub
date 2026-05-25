# turn off transformer verbose messages
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st

from xai_compare.run_explorer import RunExplorer
from xai_compare.dashboard.components.layout import apply_layout
from xai_compare.dashboard.components.run_display import display_comparable_runs

@st.cache_resource
def get_explorer():
    return RunExplorer()

explorer = get_explorer()
recent_runs = explorer.get_comparable_runs()

apply_layout("XAI Compare")

st.markdown("**Recent Datasets/Models:**")
display_comparable_runs(recent_runs)

