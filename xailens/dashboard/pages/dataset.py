import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.services import get_explorer
from xailens.dashboard.utils.render import dataset
from xailens.dashboard.utils.run_display import display_comparable_runs
from xailens import strings

def main():
    explorer = get_explorer()

    # get dataset from params and check exists
    dataset = st.query_params.get('dataset')


    apply_layout(dataset)

    # show breadcrumb


    # overall stat graphs for accuracy etc


    # instance info


    # pairwise comparison
    all_runs = explorer.get_comparable_modelevals()
    st.html(display_comparable_runs(all_runs))

if __name__ == "__main__":
    main()