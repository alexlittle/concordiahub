import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.services import get_explorer
from xailens.dashboard.utils.render import home

from xailens import strings

def main():
    explorer = get_explorer()

    all_datasets= explorer.get_datasets()

    apply_layout(strings.HOMEPAGE_TITLE)

    st.html(home.render_datasets(all_datasets))

if __name__ == "__main__":
    main()
