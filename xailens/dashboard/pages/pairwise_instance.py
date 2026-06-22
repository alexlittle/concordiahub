import streamlit as st
import plotly

from xailens.entities.context import ComparisonContext
from xailens.exceptions import InvalidConfigurationError, InvalidUrlParamsError
from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.utils.render.breadcrumb import display_breadcrumbs
from xailens.dashboard.utils.params import get_modelevals_from_url_param
from xailens.registry.comparison_registry import get_comparison_adapter
from xailens.dashboard.utils.detail_helpers import build_model_panel, render_model_panel, select_comparison_viz
from xailens import strings, config

def main():
    try:
        modeleval_a, modeleval_b = get_modelevals_from_url_param(st.query_params)
    except InvalidUrlParamsError as e:
        st.error(e)
        st.stop()

    existing_params = dict(st.query_params)
    instance_id = st.query_params.get("instance_id")
    method_a = modeleval_a.get("xai_methods")[0]
    method_b = modeleval_b.get("xai_methods")[0]

    panel_a = build_model_panel(modeleval_a, "model_a_viz", st.query_params)
    panel_b = build_model_panel(modeleval_b, "model_b_viz", st.query_params)

    try:
        comparison_class = get_comparison_adapter(method_a, method_b)
    except InvalidConfigurationError as e:
        print(e)

    # Display
    apply_layout(strings.DETAIL_TITLE)
    st.html(display_breadcrumbs(modeleval_a, modeleval_b, instance_id))
    st.subheader("**Comparison Explanation:**")

    try:
        comparison = comparison_class()
        viz_options = comparison.get_visualisations_available()
        comp_viz = st.query_params.get("comp_viz", comparison.get_default_visualisation())
        st.html(select_comparison_viz(comp_viz, viz_options, existing_params))
        ctx = ComparisonContext(
            explanation_a=panel_a.explanation,
            explanation_b=panel_b.explanation,
            modeleval_a=modeleval_a,
            modeleval_b=modeleval_b,
            axis_range = config.DEFAULT_COMPARISON_AXIS_RANGE,
        )
        display_data = comparison.render_comparison(ctx, comp_viz)
        if isinstance(display_data, plotly.graph_objs.Figure):
            st.plotly_chart(display_data, key="comparison")
        else:
            st.html(display_data)
    except (InvalidConfigurationError, NameError):
        st.warning(f"No comparison available for {method_a} vs {method_b}")
    except NotImplementedError:
        st.warning(f"{comp_viz} is not yet implemented")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        render_model_panel(panel_a, existing_params, plot_key="explanation_a")
    with col2:
        render_model_panel(panel_b, existing_params, plot_key="explanation_b")


if __name__ == "__main__":
    main()