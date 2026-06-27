import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.services import get_explorer
from xailens.dashboard.utils import home

from xailens import strings

def main():
    explorer = get_explorer()

    apply_layout(strings.HOMEPAGE_TITLE)

    st.html(home.render_datasets(explorer))

if __name__ == "__main__":
    main()
