import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.services import get_explorer
from xailens.dashboard.utils.render.breadcrumb import display_breadcrumbs
from xailens.dashboard.utils.render.dataset import (
    render_accuracy_barchart,
    render_f1_weighted_barchart,
    render_precision_weighted_barchart,
    render_recall_weighted_barchart)
from xailens.entities.dataset import Dataset

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

    # Get all the models for the dataset
    dataset_obj = Dataset(dataset, explorer.get_modelevals_for_dataset(dataset))

    # get the metrics
    metrics = dataset_obj.get_metrics()
    metrics = metrics.sort_values(by=['model'])

    # overall stat graphs for accuracy /f1
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.plotly_chart(render_accuracy_barchart(metrics))
    with col2:
        st.plotly_chart(render_f1_weighted_barchart(metrics))

    # precision/recall bar charts
    col3, col4 = st.columns(2, gap="large")
    with col3:
        st.plotly_chart(render_precision_weighted_barchart(metrics))
    with col4:
        st.plotly_chart(render_recall_weighted_barchart(metrics))

    # Global explanations

    # Instance overview


if __name__ == "__main__":
    main()