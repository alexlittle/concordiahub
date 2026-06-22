# turn off transformer verbose messages
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from xailens.dashboard.services import get_explorer
from xailens.dashboard import settings
from xailens import strings

explorer = get_explorer()

home = st.Page("pages/home.py", title=strings.HOMEPAGE_TITLE)
dataset = st.Page("pages/dataset.py", title=strings.DATASET_TITLE)
instances = st.Page("pages/instances.py", title=strings.INSTANCE_TITLE)
pairwise_overview = st.Page(f"pages/{settings.PW_OVERVIEW_FILENAME}.py", title=strings.PAIRWISE_TITLE)
pairwise_instance = st.Page(f"pages/{settings.PW_INSTANCE_FILENAME}.py", title=strings.PAIRWISE_INSTANCE_TITLE)

pg = st.navigation([home, dataset, instances, pairwise_overview, pairwise_instance], position="hidden")
pg.run()