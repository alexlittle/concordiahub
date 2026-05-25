# turn off transformer verbose messages
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from xai_compare.dashboard.services import get_explorer

explorer = get_explorer()


overview = st.Page("pages/overview.py", title="Overview")
home = st.Page("pages/home.py", title="Home")

pg = st.navigation([home, overview], position="hidden")
pg.run()