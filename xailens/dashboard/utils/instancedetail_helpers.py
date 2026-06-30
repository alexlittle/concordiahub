import uuid

import streamlit as st
import plotly
import pandas as pd

from dataclasses import dataclass

from xailens.registry.xai_registry import get_xai_adapter
from xailens.registry.viz_registry import get_viz_adapter
from xailens import exceptions
from xailens.dashboard.utils.display_table import render_dataframe
from xailens.dashboard import settings
from xailens.entities.modeleval import ModelEval
from xailens.dashboard.utils.plotly_theme import apply_horizontal_bar_theme

@dataclass
class ModelPanel:
    modeleval: ModelEval
    xai_class: type
    xai_instance: object
    viz_key: str
    viz_name: str
    viz_class: type
    explanation: object
    instance_uuid: uuid.UUID


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
                f'<a href="{settings.INSTANCE_DETAIL_FILENAME}?{query_string}" target="_self">{v.display_name}</a>'
            )
    return " | ".join(option_list)

def build_model_panel(modeleval: ModelEval, viz_param_key, viz_params, instance_uuid) -> ModelPanel:
    method = modeleval.get("xai_methods")[0]
    xai_class = get_xai_adapter(method)
    xai_instance = xai_class()
    viz_name = viz_params.get(viz_param_key, xai_instance.get_default_visualisation())

    try:
        viz_class = get_viz_adapter(viz_name)
    except (KeyError, exceptions.InvalidConfigurationError):
        viz_class = None

    explanation = xai_class.load_local_explanation(modeleval, instance_uuid)

    return ModelPanel(
        modeleval=modeleval,
        xai_class=xai_class,
        xai_instance=xai_instance,
        viz_key=viz_param_key,
        viz_name=viz_name,
        viz_class=viz_class,
        explanation=explanation,
        instance_uuid=instance_uuid
    )


def render_model_panel(panel: ModelPanel, existing_params: dict, plot_key: str):
    with st.expander(panel.modeleval.get('model_display_name'), expanded=False):

        axis_range = [-1.1, 1.1]

        viz_options = panel.xai_instance.get_compatible_visualisations()
        viz_names = list(viz_options.keys()) if isinstance(viz_options, dict) else list(viz_options)
        if panel.viz_key not in st.session_state or st.session_state[panel.viz_key] not in viz_names:
            st.session_state[panel.viz_key] = viz_names[0]
        if len(viz_names) > 1:
            st.radio("View as:",
                     options=viz_names,
                     format_func=lambda k: viz_options[k].display_name if isinstance(viz_options, dict) else k,
                     horizontal=True,
                     key=panel.viz_key,
                     label_visibility="collapsed")

        if panel.viz_class is None:
            st.warning(f"{panel.viz_name} is not a registered visualisation")
            st.html("</div>")
            return

        try:
            viz_list = panel.viz_class().render(panel.explanation, axis_range=axis_range)
            for viz in viz_list:
                if isinstance(viz, plotly.graph_objs.Figure):
                    viz = apply_horizontal_bar_theme(viz)
                    st.plotly_chart(viz, key=plot_key)
                elif isinstance(viz, pd.DataFrame):
                    st.html(render_dataframe(viz))
                else:
                    st.html(viz)
            st.html(panel.xai_instance.display_notes)
        except NotImplementedError:
            st.warning(f"{panel.viz_name} is not yet implemented for {panel.modeleval.get('model_display_name')}")
