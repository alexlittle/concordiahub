"""
Functions for the Streamlit page layout
"""


import streamlit as st

from typing import Optional
from xailens.dashboard.utils.styles import load_css
from xailens import strings, config
from xailens.dashboard.utils import sidebar


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
    selected_ds = []
    for name in sorted(all_model_names):
        if st.checkbox(name, value=name in selected_model_names, key=f"chk_{dataset_id}_{name}"):
            selected_ds.append(name)

    selected_ig = None
    if selected_instance_group:
        options = list(strings.INSTANCE_GROUPS)
        selected_ig = st.radio(
            label=f"{strings.SIDEBAR_INSTANCE_GROUP_FILTER_TITLE}",
            options=options,
            format_func=lambda k: strings.INSTANCE_GROUPS[k],
            index=options.index(selected_instance_group) if selected_instance_group in options else 0,
            key=f"radio_{dataset_id}_instance_group"
        )

    if st.button("Apply selection"):
        st.query_params["dataset"] = dataset_id
        st.query_params["models"] = ",".join(selected_ds)
        if selected_ig:
            st.query_params["instances"] = selected_ig
        st.rerun()

