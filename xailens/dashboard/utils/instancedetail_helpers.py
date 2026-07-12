import uuid

import streamlit as st
import plotly
import pandas as pd

from dataclasses import dataclass

from xailens.registry.xai_registry import get_xai_adapter
from xailens.registry.viz_registry import get_viz_adapter
from xailens import exceptions, config
from xailens.dashboard.utils.display_table import render_dataframe
from xailens.entities.model import Model
from xailens.dashboard.utils.plotly_theme import apply_horizontal_bar_theme

@dataclass
class ModelPanel:
    model: Model
    xai_class: type
    xai_instance: object
    viz_key: str
    viz_name: str
    viz_class: type
    explanation: object
    instance_uuid: uuid.UUID


def build_model_panel(model: Model, viz_param_key, viz_params, instance_uuid) -> ModelPanel:
    method = model.get("xai_methods")[0]
    xai_class = get_xai_adapter(method)
    xai_instance = xai_class()
    viz_name = viz_params.get(viz_param_key, xai_instance.get_default_visualisation())

    try:
        viz_class = get_viz_adapter(viz_name)
    except (KeyError, exceptions.InvalidConfigurationError):
        viz_class = None

    explanation = xai_class.load_local_explanation(model, instance_uuid)

    return ModelPanel(
        model=model,
        xai_class=xai_class,
        xai_instance=xai_instance,
        viz_key=viz_param_key,
        viz_name=viz_name,
        viz_class=viz_class,
        explanation=explanation,
        instance_uuid=instance_uuid
    )


def render_model_panel(panel: ModelPanel, plot_key: str):
    with st.expander(f"{panel.model.get('model_display_name')} Explanation", expanded=False):

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
                    st.plotly_chart(viz, key=plot_key, config=config.PLOTLY_CONFIG)
                elif isinstance(viz, pd.DataFrame):
                    st.html(render_dataframe(viz))
                else:
                    st.html(viz)
            st.html(panel.xai_instance.display_notes)
        except NotImplementedError:
            st.warning(f"{panel.viz_name} is not yet implemented for {panel.model.get('model_display_name')}")
