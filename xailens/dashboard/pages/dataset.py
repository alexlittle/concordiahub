import streamlit as st

from xailens.exceptions import InvalidSelectionError
from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.services import get_explorer
from xailens.dashboard.utils.render.common import display_breadcrumbs, render_metric_card
from xailens.dashboard.utils.render.dataset import (
    render_accuracy_barchart,
    render_f1_weighted_barchart,
    render_precision_weighted_barchart,
    render_recall_weighted_barchart,
    render_global_features_heatmap)
from xailens.entities.dataset import Dataset
from xailens import strings
from xailens.dashboard import settings

def main():
    explorer = get_explorer()

    # get dataset from params
    dataset = st.query_params.get('dataset')
    models_param = st.query_params.get("models")

    all_modelevals = explorer.get_modelevals_for_dataset(dataset)
    all_model_names = [me.get("model_display_name") for me in all_modelevals]

    if models_param:
        selected_model_names = models_param.split(",")
    else:
        selected_model_names = all_model_names

    apply_layout(strings.DATASET_TITLE.format(dataset=dataset),
                 all_model_names=sorted(all_model_names),
                 selected_model_names=selected_model_names,
                 dataset_id=dataset)

    if not explorer.is_valid_dataset(dataset):
        st.error(f"Dataset {dataset} does not exist")
        st.stop()

    # show breadcrumb
    st.html(display_breadcrumbs(dataset))

    # Get the selected models for the dataset
    selected_modelevals = explorer.get_selected_modelevals_for_dataset(dataset, selected_model_names)
    dataset_obj = Dataset(dataset, selected_modelevals)

    # Performance metrics
    st.html(f"<h2>{strings.PERF_METRICS_TITLE}</h2>")
    # get the metrics
    metrics = dataset_obj.get_metrics()
    metrics = metrics.sort_values(by=['model'])

    # overall stat graphs for accuracy /f1
    col1, col2 = st.columns(2, gap="large")

    col1.plotly_chart(render_accuracy_barchart(metrics))
    col2.plotly_chart(render_f1_weighted_barchart(metrics))

    # precision/recall bar charts
    col3, col4 = st.columns(2, gap="large")
    col3.plotly_chart(render_precision_weighted_barchart(metrics))
    col4.plotly_chart(render_recall_weighted_barchart(metrics))

    # Global explanations
    st.html(f"<h2>{strings.GLOBAL_EXPLANATION_TITLE}</h2>")
    try:
        df_global_exp = dataset_obj.get_global_explanations()
        st.plotly_chart(render_global_features_heatmap(df_global_exp), width='content')
        st.html("* Global feature importance is not available for LLMs.")
    except InvalidSelectionError:
        st.html("Global feature importance is not displayed when only LLMs are selected.")


    # Instance overview
    st.html(f"<h2>{strings.INSTANCE_CONCENSUS_TITLE}</h2>")


    instance_metrics = dataset_obj.get_instance_consensus()
    col1, col2, col3, col4 = st.columns(4)
    params = st.query_params.to_dict()
    query_string = "&".join(f"{k}={v}" for k, v in params.items())

    col1.html(render_metric_card(strings.INSTANCES_TOTAL_TITLE,
                                 instance_metrics['total'],
                                 f"{settings.INSTANCES_FILENAME}?{query_string}&instances=all"))
    col2.html(render_metric_card(strings.INSTANCES_ALL_CORRECT_TITLE,
                                 instance_metrics['all_correct'],
                                 f"{settings.INSTANCES_FILENAME}?{query_string}&instances=allcorrect"))
    col3.html(render_metric_card(strings.INSTANCES_ALL_INCORRECT_TITLE,
                                 instance_metrics['all_incorrect'],
                                 f"{settings.INSTANCES_FILENAME}?{query_string}&instances=allincorrect"))
    col4.html(render_metric_card(strings.INSTANCES_DISAGREE_TITLE,
                                 instance_metrics['models_disagree'],
                                 f"{settings.INSTANCES_FILENAME}?{query_string}&instances=disagree"))

if __name__ == "__main__":
    main()