"""
Streamlit homepage - to select a dataset
"""

import streamlit as st

from concordiahub.dashboard.utils.layout import apply_layout
from concordiahub.dashboard.explorer import get_explorer
from concordiahub.dashboard.utils import home_helpers

from concordiahub import strings

def main():
    """
    Streamlit homepage - to select a dataset
    """
    explorer = get_explorer()

    if explorer.get_datasets():
        apply_layout(strings.HOMEPAGE_TITLE)
        st.html(home_helpers.render_datasets(explorer))
    else:
        apply_layout(strings.HOMEPAGE_NO_DATASETS_TITLE)
        st.html(strings.HOMEPAGE_NO_DATASETS_DESC)

if __name__ == "__main__":
    main()
