import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.services import get_explorer
from xailens.dashboard.utils.render import dataset

from xailens import strings

def main():
    explorer = get_explorer()

    # get dataset from params and check exists


    apply_layout(strings.DATASET_TITLE)

    # show breadcrumb


    # overall stat graphs for accuracy etc


    # instance info


    # pairwise comparison


if __name__ == "__main__":
    main()