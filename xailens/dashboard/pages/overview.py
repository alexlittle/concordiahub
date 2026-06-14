
import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.services.comparison import build_comparison_df
from xailens.exceptions import  InvalidUrlParamsError
from xailens.dashboard.utils.params import get_runs_from_url_param
from xailens.dashboard.utils.comparison import generate_comparison_table_html, render_run_summary
from xailens.dashboard.utils.breadcrumb import display_breadcrumbs
from xailens import strings

def main():
    # Get/load the necessary data for this page
    try:
        run_a, run_b = get_runs_from_url_param(st.query_params)
    except InvalidUrlParamsError as e:
        st.error(e)
        st.stop()

    df = build_comparison_df(run_a=run_a, run_b=run_b)

    # Page layout/display
    apply_layout(strings.OVERVIEW_TITLE)

    st.html(display_breadcrumbs(run_a, run_b))

    col1, col2 = st.columns(2)

    with col1:
        render_run_summary(run_a)

    with col2:
        render_run_summary(run_b)

    st.markdown("**Test samples available for comparison:**")
    table_html = generate_comparison_table_html(df, run_a, run_b)
    st.html(table_html)

if __name__ == "__main__":
    main()









