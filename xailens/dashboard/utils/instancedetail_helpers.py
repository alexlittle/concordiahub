"""
Helper functions for instance detail page
"""

import uuid

import streamlit as st
import plotly
import pandas as pd

from typing import MutableMapping, Any
from dataclasses import dataclass

from xailens.registry.xai_registry import get_xai_adapter
from xailens.registry.viz_registry import get_viz_adapter
from xailens import exceptions, config
from xailens.dashboard.utils.display_table import dataframe_to_html_table
from xailens.entities.model import Model
from xailens.dashboard.utils.plotly_theme import apply_horizontal_bar_theme
from xailens.dashboard.utils.common import title_with_help_icon

@dataclass
class ModelPanel:
    """
    Dataclass representing a model panel for a model explanation
    """
    model: Model
    xai_class: type
    xai_instance: object
    viz_key: str
    viz_name: str
    viz_class: type
    explanation: object
    instance_uuid: uuid.UUID


def build_model_panel(
        model: Model,
        viz_param_key: str,
        viz_params: MutableMapping[str, Any],
        instance_uuid:str) -> ModelPanel:
    """
    Builds a ModelPanel to display the model's explanation

    Args:
        model (Model): Model object
        viz_param_key (str): key identifying the visualization parameters
        viz_params (MutableMapping[str, Any]): Streamlit session state (or
            dict-like equivalent) containing visualization parameters
        instance_uuid (str): identifier for the instance being visualized

    Returns:
        ModelPanel: panel object built from the model and visualization params
    """

    # get the model/XAI method and the default viz name
    method = model.get("xai_methods")[0]
    xai_class = get_xai_adapter(method)
    xai_instance = xai_class()
    viz_name = viz_params.get(viz_param_key,
                              xai_instance.get_default_visualization())

    # instantiate the viz adapter
    try:
        viz_class = get_viz_adapter(viz_name)
    except (KeyError, exceptions.InvalidConfigurationError):
        viz_class = None

    # load the explanations for the model and instance
    explanation = xai_class.load_local_explanation(model, instance_uuid)

    # return a populated ModelPanel instance
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


def render_model_panel(panel: ModelPanel, plot_key: str) -> None:
    """
    Renders a ModelPanel from the given panel

    Args:
        panel (ModelPanel): ModelPanel object
        plot_key (str): unique identifier for the plot

    """
    with st.container(border=True, key=f"panel-{plot_key}"):

        # set title and help icon
        title = title_with_help_icon(
            f"{panel.model.get('model_display_name')} Explanation",
            panel.xai_instance.display_notes)
        st.html(title)

        # fix axis range - always normalized -1 to 1 for these, so a little
        # extra for display
        axis_range = [-1.1, 1.1]

        # get the viz options for current model/XAI method combination
        viz_options = panel.xai_instance.get_compatible_visualizations()
        viz_names = list(viz_options.keys()) if isinstance(viz_options, dict) else list(viz_options)

        # set session state for current panel display type
        if panel.viz_key not in st.session_state or st.session_state[panel.viz_key] not in viz_names:
            st.session_state[panel.viz_key] = viz_names[0]

        # show display options, if more than one available
        if len(viz_names) > 1:
            st.radio("View as:",
                     options=viz_names,
                     format_func=lambda k: viz_options[k].display_name if isinstance(viz_options, dict) else k,
                     horizontal=True,
                     key=panel.viz_key,
                     label_visibility="collapsed")

        # Check in case invalid viz is requested
        if panel.viz_class is None:
            st.warning(f"{panel.viz_name} is not a registered visualization")
            return

        # show the individual parts of the explanation. In most cases there is
        # only one part, but with LLM explanations, there is the overall
        # explanation and tabular format for each feature
        try:
            viz_list = panel.viz_class().render(panel.explanation,
                                                axis_range=axis_range)
            for viz in viz_list:
                # Decide how to display based on object type
                if isinstance(viz, plotly.graph_objs.Figure):
                    viz = apply_horizontal_bar_theme(viz)
                    st.plotly_chart(viz,
                                    key=plot_key,
                                    config=config.PLOTLY_CONFIG)
                elif isinstance(viz, pd.DataFrame):
                    st.html(dataframe_to_html_table(viz))
                else:
                    st.html(viz)
        except NotImplementedError:
            # clean failure if viz requested is not yet implemented
            st.warning(f"{panel.viz_name} is not yet implemented for "
                       f"{panel.model.get('model_display_name')}")
