"""
Starts the Streamlit app
"""

import os

# turn off transformer verbose messages
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from concordiahub.dashboard.explorer import get_explorer
from concordiahub.dashboard import settings
from concordiahub import strings

explorer = get_explorer()

home = st.Page("pages/home.py", title=strings.HOMEPAGE_TITLE)
dataset = st.Page("pages/dataset.py", title=strings.DATASET_TITLE)
instances = st.Page("pages/instances.py", title=strings.INSTANCE_TITLE)
instance_detail = st.Page(f"pages/{settings.INSTANCE_DETAIL_FILENAME}.py", title=strings.INSTANCE_DETAIL_TITLE)

pg = st.navigation([home, dataset, instances, instance_detail], position="hidden")
pg.run()