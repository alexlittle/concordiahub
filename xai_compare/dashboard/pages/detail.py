import streamlit as st

from xai_compare.dashboard.components.layout import apply_layout
from xai_compare.dashboard.services import get_explorer
from xai_compare.dashboard.components.breadcrumb import display_breadcrumbs

explorer = get_explorer()

# get the runs from the parameters
runs_param = st.query_params.get("runs")
if not runs_param:
    st.warning("No runs selected")
    st.stop()
run_ids = runs_param.split(",")

runs = explorer.get_runs_by_uuids(run_ids)

if len(runs) != 2:
    st.error("You need exactly 2 runs to compare")
    st.stop()

run_a, run_b = runs

# get the sample id from the parameters
sample_id = st.query_params.get("sample_id")


# Disaply page
apply_layout("XAI Compare")

st.html(display_breadcrumbs(run_a, run_b, sample_id))

st.subheader("**Comparison Explanation:**")


col1, col2 = st.columns(2)

with col1:
    st.subheader("**Run A Explanation:**")

with col2:
    st.subheader("**Run B Explanation:**")