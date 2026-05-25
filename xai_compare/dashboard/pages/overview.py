
import streamlit as st

from xai_compare.dashboard.components.layout import apply_layout
from xai_compare.dashboard.services import get_explorer


apply_layout("XAI Compare")

runs_param = st.query_params.get("runs")

if not runs_param:
    st.warning("No runs selected")
    st.stop()

run_ids = runs_param.split(",")

st.write("Selected runs:", run_ids)

explorer = get_explorer()

explorer.get_runs_by_uuids(run_ids)
