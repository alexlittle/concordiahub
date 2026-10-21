"""
Functions for the Streamlit page layout
"""


import streamlit as st

from typing import Optional
from concordiahub.dashboard.utils.styles import load_css
from concordiahub import strings, config
from concordiahub.dashboard.utils import sidebar


def apply_layout(title: str,
                 dataset_id: Optional[str] = None,
                 all_model_names: Optional[list] = None,
                 selected_model_names: Optional[list] = None,
                 selected_instance_group: Optional[str] = None,):
    """
    Applies the layout to the Streamlit page

    Args:
        title (str): title of the page
        dataset_id (Optional[str]): current dataset id, or None
        all_model_names (Optional[list]): all the models for the current dataset, or None
        selected_model_names (Optional[list]): the selected models for the current dataset, or None
        selected_instance_group (Optional[str]): the selected instance group for the current dataset, or None

    """

    # Page title
    st.set_page_config(page_title=f"{strings.APP_NAME} - {title}", layout="wide")

    # inject css
    load_css()

    # Sidebar
    with st.sidebar:
        st.html(sidebar.render_header())
        if all_model_names:
            render_sidebar_options(all_model_names,
                                   selected_model_names,
                                   dataset_id,
                                   selected_instance_group)

        st.html(sidebar.render_footer())

    # Page title
    st.html(f"<h1 class='header'>{title}</h1>")


def render_sidebar_options(all_model_names: list,
                           selected_model_names: list,
                           dataset_id: Optional[str] = None,
                           selected_instance_group: Optional[str] = None,):
    """
    Renders the sidebar options and filter form for the current dataset

    Args:
        all_model_names (list): all the models for the current dataset
        selected_model_names (list): selected models for the current dataset
        dataset_id (Optional[str]): current dataset id, or None
        selected_instance_group (Optional[str]): selected instance group for the current dataset, or None

    """

    st.html(f"<h3>{strings.SIDEBAR_MODEL_FILTER_TITLE}</h3>")

    chk_keys = {n: f"chk_{dataset_id}_{n}" for n in sorted(all_model_names)}
    radio_key = f"radio_{dataset_id}_instance_group"

    def _apply():
        """
        Internal helper function to apply the layout to the sidebar
        """
        selected = [n for n, k in chk_keys.items() if
                    st.session_state.get(k)]
        st.query_params["dataset"] = dataset_id
        if selected:
            st.query_params["models"] = ",".join(selected)
        else:
            st.query_params.pop("models", None)
        if selected_instance_group:
            st.query_params["instances"] = st.session_state[radio_key]

    with st.form(f"filters_{dataset_id}", border=False):
        for name, k in chk_keys.items():
            st.checkbox(name, value=name in selected_model_names, key=k)

        if selected_instance_group:
            options = list(strings.INSTANCE_GROUPS)
            st.radio(
                label=strings.SIDEBAR_INSTANCE_GROUP_FILTER_TITLE,
                options=options,
                format_func=lambda k: strings.INSTANCE_GROUPS[k],
                index=options.index(selected_instance_group)
                if selected_instance_group in options else 0,
                key=radio_key,
            )

        st.form_submit_button("Apply selection", on_click=_apply)

