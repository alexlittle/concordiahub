import streamlit as st

from xailens.exceptions import InvalidSelectionError
from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.explorer import get_explorer
from xailens.dashboard.utils.common import display_breadcrumbs, render_metric_card
from xailens.dashboard.utils.dataset_helpers import render_metric_barchart
from xailens.dashboard.utils.heatmap import render_features_heatmap
from xailens.entities.dataset import Dataset
from xailens import strings, config
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

    with st.container(border=True):
        recommended_models = dataset_obj.get_recommended_models()

        if len(recommended_models) == 1:
            st.html(f"<h3>{strings.RECOMMENDED_MODEL_TITLE}: {recommended_models[0]}</h3>")
        else:
            st.html(f"<h3>{strings.RECOMMENDED_MODELS_TITLE}: {', '.join(recommended_models)}</h3>")

    # Get performance metrics
    metrics = dataset_obj.get_metrics()
    metrics = metrics.sort_values(by=['model'])

    # Show f1 weighted and global explanations
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.html(f"<h3>{strings.PERF_FI_WEIGHTED_TITLE}</h3>")
            st.plotly_chart(render_metric_barchart(metrics, "f1_weighted", show_title=False))
    with col2:
        with st.container(border=True):
            # Global explanations
            st.html(f"<h3>{strings.GLOBAL_EXPLANATION_TITLE}</h3>")
            try:
                df_global_exp = dataset_obj.get_global_explanations()
                st.plotly_chart(render_features_heatmap(df_global_exp),
                                width='content')
                st.html(
                    "* Global feature importance is not available for LLMs.")
            except InvalidSelectionError:
                st.html(
                    "Global feature importance is not displayed when only LLMs are selected.")


    # accuracy/precision/recall bar charts
    with st.expander(f"{strings.PERF_METRICS_TITLE}", expanded=False):
        col3, col4, col5 = st.columns(3)
        with col3:
            with st.container(border=True):
                st.plotly_chart(render_metric_barchart(metrics, "accuracy"))
        with col4:
            with st.container(border=True):
                st.plotly_chart(render_metric_barchart(metrics, "precision_weighted"))
        with col5:
            with st.container(border=True):
                st.plotly_chart(
                    render_metric_barchart(metrics, "recall_weighted"))




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