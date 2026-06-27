
import streamlit as st

from xailens.entities.model_explorer import ModelExplorer

@st.cache_resource
def get_explorer():
    return ModelExplorer()
