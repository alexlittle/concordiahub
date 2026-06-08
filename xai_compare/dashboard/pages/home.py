import streamlit as st

from xai_compare.dashboard.utils.layout import apply_layout
from xai_compare.dashboard.utils.run_display import display_comparable_runs
from xai_compare.dashboard.services import get_explorer

from xai_compare import strings

def main():
    explorer = get_explorer()

    recent_runs = explorer.get_comparable_runs(10)

    apply_layout(strings.APP_NAME)

    st.markdown("**Recent Datasets/Models:**")
    display_comparable_runs(recent_runs)

if __name__ == "__main__":
    main()
