import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.services import get_explorer
from xailens.dashboard.utils.render import dataset
from xailens.dashboard.utils.run_display import display_comparable_runs
from xailens.dashboard.utils.filter import apply_dataset_filter
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
    filtered_runs = apply_dataset_filter(all_runs, dataset)
    st.html(display_comparable_runs(filtered_runs))

if __name__ == "__main__":
    main()