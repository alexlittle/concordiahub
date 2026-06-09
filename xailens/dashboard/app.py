# turn off transformer verbose messages
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from xailens.dashboard.services import get_explorer

explorer = get_explorer()


overview = st.Page("pages/overview.py", title="Overview")
home = st.Page("pages/home.py", title="Home")
detail = st.Page("pages/detail.py", title="Detail")

pg = st.navigation([home, overview, detail], position="hidden")
pg.run()