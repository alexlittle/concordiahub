import streamlit as st

from xai_compare.exceptions import InvalidConfigurationError
from xai_compare.dashboard.components.formatting import format_model_name
from xai_compare.dashboard.components.layout import apply_layout
from xai_compare.dashboard.services import get_explorer
from xai_compare.dashboard.components.breadcrumb import display_breadcrumbs

from xai_compare.registry.xai_registry import get_xai_adapter
from xai_compare.registry.viz_registry import get_viz_adapter
from xai_compare.registry.comparison_registry import get_comparison_adapter
from xai_compare import constants, strings

explorer = get_explorer()

# get the runs from the parameters
runs_param = st.query_params.get("runs")
if not runs_param:
    st.warning(strings.ERR_NO_RUNS)
    st.stop()
run_ids = runs_param.split(",")

runs = explorer.get_runs_by_uuids(run_ids)

if len(runs) != 2:
    st.error(strings.ERR_NOT_2_RUNS)
    st.stop()

run_a, run_b = runs

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

comparison_class = get_comparison_adapter(method_a, method_b)
xai_a = get_xai_adapter(method_a)
xai_b = get_xai_adapter(method_b)
exp_a = xai_a.load_explanation(run_a, sample_id)
exp_b = xai_b.load_explanation(run_b, sample_id)

comp_max = comparison_class.get_abs_max(exp_a, exp_b, run_a.get(constants.DATA_SCHEMA_ID_COLUMN))

abs_max = max(
    xai_class_a.get_abs_max(exp_a, run_a.get(constants.DATA_SCHEMA_ID_COLUMN)),
    xai_class_b.get_abs_max(exp_b, run_a.get(constants.DATA_SCHEMA_ID_COLUMN)),
    comp_max
)
margin = abs_max * 0.15
axis_range = [-(abs_max + margin), abs_max + margin]

# Display page
apply_layout(strings.APP_NAME)

st.html(display_breadcrumbs(run_a, run_b, sample_id))

st.subheader("**Comparison Explanation:**")


try:
    fig = comparison_class().render_comparison(exp_a, exp_b, run_a, run_b, axis_range=axis_range)
    st.plotly_chart(fig, key="comparison")
except InvalidConfigurationError:
    st.warning(f"No comparison available for {method_a} vs {method_b}")


col1, col2 = st.columns(2)

with col1:
    st.subheader(f"**{format_model_name(run_a)} Explanation:**")
    fig = viz_class_a().render_local(explanation_a, run_a.get(constants.DATA_SCHEMA_ID_COLUMN), axis_range=axis_range)
    st.plotly_chart(fig, key="explanation_a")

with col2:
    st.subheader(f"**{format_model_name(run_b)} Explanation:**")
    fig = viz_class_b().render_local(explanation_b, run_b.get(constants.DATA_SCHEMA_ID_COLUMN), axis_range=axis_range)
    st.plotly_chart(fig, key="explanation_b")