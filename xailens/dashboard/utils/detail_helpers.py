import streamlit as st
import plotly

from dataclasses import dataclass

from xailens.registry.xai_registry import get_xai_adapter
from xailens.registry.viz_registry import get_viz_adapter
from xailens import constants, exceptions
from xailens.dashboard.utils.formatting import format_model_name


@dataclass
class ModelPanel:
    run: dict
    xai_class: type
    xai_instance: object
    viz_key: str       # e.g. "model_a_viz"
    viz_name: str      # e.g. "bar"
    viz_class: type
    explanation: object

def select_individual_viz(viz_id: str, current: str, options: dict, existing_params: dict) -> str:
    option_list = []
    for k, v in options.items():
        if k == current:
            option_list.append(f"<strong>{v.display_name}</strong>")
        else:
            link_params = {**existing_params, viz_id: k}
            query_string = "&".join(
                f"{pk}={pv}" for pk, pv in link_params.items())
            option_list.append(
                f'<a href="detail?{query_string}" target="_self">{v.display_name}</a>'
            )
    return " | ".join(option_list)

def select_comparison_viz(current: str, options: dict, existing_params: dict) -> str:
    option_list =[]
    for k,v in options.items():
        if k == current:
            option_list.append(f"<strong>{v.display_name}</strong>")
        else:
            link_params = {**existing_params, "comp_viz": k}
            query_string = "&".join(f"{pk}={pv}" for pk, pv in link_params.items())
            option_list.append(
                f'<a href="detail?{query_string}" target="_self">{v.display_name}</a>'
            )
    return " | ".join(option_list)


def build_model_panel(run, viz_param_key, query_params) -> ModelPanel:
    method = run.get("xai_methods")[0]
    xai_class = get_xai_adapter(method)
    xai_instance = xai_class()
    viz_name = query_params.get(viz_param_key, xai_instance.get_default_visualisation())

    try:
        viz_class = get_viz_adapter(viz_name)
    except (KeyError, exceptions.InvalidConfigurationError):
        viz_class = None


    sample_id = query_params.get("sample_id")
    explanation = xai_class.load_explanation(run, sample_id)
    return ModelPanel(
        run=run,
        xai_class=xai_class,
        xai_instance=xai_instance,
        viz_key=viz_param_key,
        viz_name=viz_name,
        viz_class=viz_class,
        explanation=explanation,
    )


def render_model_panel(panel: ModelPanel, existing_params: dict, plot_key: str):
    st.subheader(f"**{format_model_name(panel.run)} Explanation:**")
    axis_range = [-1.1, 1.1]

    viz_options = panel.xai_instance.get_compatible_visualisations()
    select_text = select_individual_viz(panel.viz_key, panel.viz_name, viz_options, existing_params)
    st.html(select_text)

    if panel.viz_class is None:
        st.warning(f"{panel.viz_name} is not a registered visualisation")
        return

    try:
        fig = panel.viz_class().render_local(
            panel.explanation,
            axis_range=axis_range,
        )
        if isinstance(fig, plotly.graph_objs.Figure):
            st.plotly_chart(fig, key=plot_key)
        else:
            st.html(fig)
    except NotImplementedError:
        st.warning(f"{panel.viz_name} is not yet implemented for {format_model_name(panel.run)}")