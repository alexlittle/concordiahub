import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.services import get_explorer
from xailens.dashboard.utils.run_display import display_comparable_runs
from xailens.dashboard.utils.filter import apply_dataset_filter
from xailens import strings
from xailens.dashboard.utils.render.breadcrumb import display_breadcrumbs

def main():
    explorer = get_explorer()


    # get dataset from params
    dataset = st.query_params.get('dataset')
    apply_layout(dataset)

    if not explorer.is_valid_dataset(dataset):
        st.error(f"Dataset {dataset} does not exist")
        st.stop()

    # show breadcrumb
    st.html(display_breadcrumbs(dataset))

    # get the
    # overall stat graphs for accuracy etc


    # instance info


    # pairwise comparison
    all_runs = explorer.get_comparable_modelevals()
    filtered_runs = apply_dataset_filter(all_runs, dataset)
    st.html(display_comparable_runs(filtered_runs))

if __name__ == "__main__":
    main()