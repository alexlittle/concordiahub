import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.explorer import get_explorer
from xailens.dashboard.utils import home_helpers

from xailens import strings

def main():
    explorer = get_explorer()

    apply_layout(strings.HOMEPAGE_TITLE)

    st.html(home_helpers.render_datasets(explorer))

if __name__ == "__main__":
    main()
