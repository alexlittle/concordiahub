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
from xailens import strings

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

    # Get all the models for the dataset
    selected_modelevals = explorer. get_selected_modelevals_for_dataset(dataset, selected_model_names)
    dataset_obj = Dataset(dataset, selected_modelevals)

    # Performance metrics
    st.html(f"<h2>{strings.PERF_METRICS_TITLE}</h2>")
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
    st.html(f"<h2>{strings.GLOBAL_EXPLANATION_TITLE}</h2>")

    df_global_exp = dataset_obj.get_global_explanations()
    responded: st.st.dataframe(df_global_exp)



    # Instance overview
    st.html(f"<h2>{strings.INSTANCE_CONCENSUS_TITLE}</h2>")


    instance_metrics = dataset_obj.get_instance_consensus()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(strings.INSTANCES_TOTAL_TITLE, instance_metrics['total'])
    col2.metric(strings.INSTANCES_ALL_CORRECT_TITLE, instance_metrics['all_correct'])
    col3.metric(strings.INSTANCES_ALL_INCORRECT_TITLE, instance_metrics['all_incorrect'])
    col4.metric(strings.INSTANCES_DISAGREE_TITLE, instance_metrics['models_disagree'])

if __name__ == "__main__":
    main()