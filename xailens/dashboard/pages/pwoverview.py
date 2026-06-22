
import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.utils.render.breadcrumb import display_breadcrumbs
from xailens.services.comparison import build_comparison_df
from xailens.exceptions import  InvalidUrlParamsError
from xailens.dashboard.utils.params import get_modelevals_from_url_param
from xailens.dashboard.utils.comparison import generate_comparison_table_html, render_modeleval_summary
from xailens import strings

def main():
    # Get/load the necessary data for this page
    try:
        modeleval_a, modeleval_b = get_modelevals_from_url_param(st.query_params)
    except InvalidUrlParamsError as e:
        st.error(e)
        st.stop()

    df = build_comparison_df(modeleval_a=modeleval_a, modeleval_b=modeleval_b)

    # Page layout/display
    apply_layout(strings.OVERVIEW_TITLE)

    st.html(display_breadcrumbs(modeleval_a, modeleval_b))

    col1, col2 = st.columns(2, gap="large")

    with col1:
        render_modeleval_summary(modeleval_a)

    with col2:
        render_modeleval_summary(modeleval_b)

    st.markdown("**Test instances available for comparison:**")
    table_html = generate_comparison_table_html(df, modeleval_a, modeleval_b)
    st.html(table_html)

if __name__ == "__main__":
    main()









