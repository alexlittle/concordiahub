
import streamlit as st

from xai_compare.dashboard.components.layout import apply_layout
from xai_compare.dashboard.services import get_explorer
from xai_compare.services.comparison import build_comparison_df
from xai_compare.dashboard.logic.comparison import generate_comparison_table_html, render_run_summary
from xai_compare.dashboard.components.breadcrumb import display_breadcrumbs
from xai_compare import strings

# Get/load the necessary data for this page
explorer = get_explorer()

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

df = build_comparison_df(run_a=run_a, run_b=run_b)

# Page layout/display
apply_layout(strings.APP_NAME)

st.html(display_breadcrumbs(run_a, run_b))


st.subheader(f"**Dataset:** {run_a.get('dataset_id')}")
col1, col2 = st.columns(2)

with col1:
    render_run_summary(run_a)

with col2:
    render_run_summary(run_b)

st.markdown("**Test samples available for comparison:**")
table_html = generate_comparison_table_html(df, run_a, run_b, runs_param)
st.html(table_html)











