import streamlit as st

from xai_compare.dashboard.components.formatting import format_model_name
from xai_compare.dashboard.components.layout import apply_layout
from xai_compare.dashboard.services import get_explorer
from xai_compare.dashboard.components.breadcrumb import display_breadcrumbs

from xai_compare.registry.xai_registry import get_xai_adapter
from xai_compare.registry.viz_registry import get_viz_adapter
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


# Display page
apply_layout(strings.APP_NAME)

st.html(display_breadcrumbs(run_a, run_b, sample_id))

st.subheader("**Comparison Explanation:**")


col1, col2 = st.columns(2)
for col, run in [(col1, run_a), (col2, run_b)]:
    with col:
        st.subheader(f"**{format_model_name(run)} Explanation:**")
        xai_method = run.get("xai_methods")[0]
        xai_class = get_xai_adapter(xai_method)
        viz_class = get_viz_adapter(xai_class.default_visualisation)
        explanation = xai_class.load_explanation(run, sample_id)
        fig = viz_class().render_local(explanation, run.get(constants.DATA_SCHEMA_ID_COLUMN))
        st.plotly_chart(fig)