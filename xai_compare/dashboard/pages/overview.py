
import streamlit as st

from xai_compare.dashboard.components.layout import apply_layout
from xai_compare.services.comparison import build_comparison_df
from xai_compare.exceptions import  InvalidUrlParamsError
from xai_compare.dashboard.utils.params import get_runs_from_url_param
from xai_compare.dashboard.utils.comparison import generate_comparison_table_html, render_run_summary
from xai_compare.dashboard.components.breadcrumb import display_breadcrumbs
from xai_compare import strings

# Get/load the necessary data for this page
try:
    run_a, run_b = get_runs_from_url_param(st.query_params)
except InvalidUrlParamsError as e:
    st.error(e)
    st.stop()

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
table_html = generate_comparison_table_html(df, run_a, run_b)
st.html(table_html)











