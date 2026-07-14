"""
Model Explorer manager - cached so not rebuilt on every page load
"""


import streamlit as st

from xailens.entities.model_explorer import ModelExplorer

@st.cache_resource
def get_explorer() -> ModelExplorer:
    """
    Helper method to get the cached ModelExplorer object

    Returns:
        ModelExplorer: ModelExplorer object
    """
    return ModelExplorer()
