import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.utils.run_display import display_comparable_runs
from xailens.dashboard.utils.filter import build_filter_pills_html, apply_pill_filter
from xailens.dashboard.services import get_explorer

from xailens import strings

def main():
    explorer = get_explorer()

    all_runs = explorer.get_comparable_runs()

    apply_layout(strings.HOMEPAGE_TITLE)

    st.html(strings.HOMEPAGE_HELPER)

    active_pill = st.query_params.get("pill_filter", "all")

    pill_options = explorer.get_run_filter_options()
    st.html(build_filter_pills_html(active=active_pill, pill_options=pill_options))

    filtered_runs = apply_pill_filter(all_runs, active_pill)
    st.html(display_comparable_runs(filtered_runs))

if __name__ == "__main__":
    main()
