import streamlit as st

from xai_compare.context import ComparisonContext
from xai_compare.exceptions import InvalidConfigurationError, InvalidUrlParamsError
from xai_compare.dashboard.utils.formatting import format_model_name
from xai_compare.dashboard.utils.layout import apply_layout
from xai_compare.dashboard.utils.breadcrumb import display_breadcrumbs
from xai_compare.dashboard.utils.params import get_runs_from_url_param
from xai_compare.registry.xai_registry import get_xai_adapter
from xai_compare.registry.viz_registry import get_viz_adapter
from xai_compare.registry.comparison_registry import get_comparison_adapter
from xai_compare import constants, strings
from xai_compare.services.preprocess import normalise_by_max
from xai_compare.dashboard.utils.viz import select_comparison_viz, select_individual_viz

# get the runs from the parameters
try:
    run_a, run_b = get_runs_from_url_param(st.query_params)
except InvalidUrlParamsError as e:
    st.error(e)
    st.stop()

# get the sample id from the parameters
sample_id = st.query_params.get("sample_id")

method_a = run_a.get("xai_methods")[0]
method_b = run_b.get("xai_methods")[0]

xai_class_a = get_xai_adapter(method_a)
viz_class_a = get_viz_adapter(xai_class_a.default_visualisation)
explanation_a = xai_class_a.load_explanation(run_a, sample_id)

xai_class_b = get_xai_adapter(method_b)
viz_class_b = get_viz_adapter(xai_class_b.default_visualisation)
explanation_b = xai_class_b.load_explanation(run_b, sample_id)

axis_range = None

try:
    comparison_class = get_comparison_adapter(method_a, method_b)
    xai_a = get_xai_adapter(method_a)
    xai_b = get_xai_adapter(method_b)

    comp_max = comparison_class.get_abs_max(explanation_a, explanation_b, run_a.get(constants.DATA_SCHEMA_ID_COLUMN))

    margin = comp_max * 0.15 # add some tolerance
    axis_range = [-(comp_max + margin), comp_max + margin]
except InvalidConfigurationError as e:
    pass

# Display page
apply_layout(strings.APP_NAME)

st.html(display_breadcrumbs(run_a, run_b, sample_id))

st.subheader("**Comparison Explanation:**")

kwargs = {}
if axis_range is not None:
    kwargs['axis_range'] = axis_range

try:
    comparison = comparison_class()
    viz_options = comparison.get_visualisations_available()
    comp_viz = st.query_params.get("comp_viz",
                                   comparison.get_default_visualisation())
    existing_params = dict(st.query_params)
    select_text = select_comparison_viz(comp_viz, viz_options, existing_params)
    st.html(select_text)

    ctx = ComparisonContext(
        explanation_a=explanation_a,
        explanation_b=explanation_b,
        run_a=run_a,
        run_b=run_b,
        **kwargs)
    fig = comparison.render_comparison(ctx, comp_viz)
    st.plotly_chart(fig, key="comparison")

except (InvalidConfigurationError, NameError) as e:
    st.warning(f"No comparison available for {method_a} vs {method_b}")
except NotImplementedError as e:
    st.warning(f"{comp_viz} is not yet implemented")


col1, col2 = st.columns(2)

with col1:
    st.subheader(f"**{format_model_name(run_a)} Explanation:**")
    normed_a = normalise_by_max(explanation_a, run_a.get(constants.DATA_SCHEMA_ID_COLUMN))
    normed_a = normed_a.sort_values(ascending=True)
    axis_range = [-1.1, 1.1]
    fig = viz_class_a().render_local(normed_a, run_a.get(constants.DATA_SCHEMA_ID_COLUMN), axis_range=axis_range)
    st.plotly_chart(fig, key="explanation_a")

with col2:
    st.subheader(f"**{format_model_name(run_b)} Explanation:**")
    normed_b = normalise_by_max(explanation_b, run_a.get(constants.DATA_SCHEMA_ID_COLUMN))
    normed_b = normed_b.sort_values(ascending=True)
    axis_range = [-1.1, 1.1]
    fig = viz_class_b().render_local(normed_b, run_b.get(constants.DATA_SCHEMA_ID_COLUMN), axis_range=axis_range)
    st.plotly_chart(fig, key="explanation_b")