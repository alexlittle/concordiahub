"""
Streamlit page to display dataset overview
"""


import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.explorer import get_explorer
from xailens.dashboard.utils.common import display_breadcrumbs, title_with_help_icon
from xailens.dashboard.utils.dataset_helpers import render_metric_barchart, render_metric_card
from xailens.dashboard.utils.upset import render_upset_figure
from xailens.dashboard.utils.heatmap import render_features_heatmap
from xailens.entities.dataset import Dataset
from xailens import strings, config
from xailens.dashboard import settings


def main():
    """
    Streamlit page to display dataset overview
    """

    explorer = get_explorer()


    # get dataset from params
    dataset = st.query_params.get('dataset')
    models_param = st.query_params.get("models")

    all_models = explorer.get_models_for_dataset(dataset)
    all_model_names = [me.get("model_display_name") for me in all_models]

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

    # show breadcrumb and anchor links

    st.html(display_breadcrumbs(dataset))

    # Get the selected models for the dataset
    selected_models = explorer.get_selected_models_for_dataset(dataset, selected_model_names)
    dataset_obj = Dataset(dataset, selected_models)

    tab1, tab2, tab3 = st.tabs(["Overview", "More Metrics", "Instance Consensus"])

    with tab1:
        with st.container(border=True, key="recommended-models"):
            recommended_models = dataset_obj.get_recommended_models()
            if len(recommended_models) == 1:
                title = f"""
                    <span class='recommended-title'>{strings.RECOMMENDED_MODEL_TITLE}</span>: 
                    <span class='recommended-model'>{recommended_models[0]}"""
            else:
                title = f"""
                        <span class='recommended-title'>{strings.RECOMMENDED_MODELS_TITLE}</span>: 
                        <span class='recommended-model'>{', '.join(recommended_models)}</span>"""

            st.html(title_with_help_icon(title, strings.HELP_RECOMMENDED_MODEL))

        # Get performance metrics
        metrics = dataset_obj.get_metrics()
        metrics = metrics.sort_values(by=['model'])

        col1, col2= st.columns([2, 2])
        with col1:
            # F1 weighted score
            with st.container(border=True):
                st.html(title_with_help_icon(strings.METRICS_NAMES["f1_weighted"],
                                             strings.HELP_METRICS["f1_weighted"]))
                st.plotly_chart(render_metric_barchart(metrics, "f1_weighted", show_title=False), config=config.PLOTLY_CONFIG)
        with col2:
            # Global explanations
            with st.container(border=True):
                st.html(title_with_help_icon(strings.GLOBAL_EXPLANATION_TITLE, strings.HELP_GLOBAL_EXPLANATION))
                df_global_exp = dataset_obj.get_global_explanations()
                st.plotly_chart(render_features_heatmap(df_global_exp), width='content', config=config.PLOTLY_CONFIG)

    with tab2:
        # accuracy/precision/recall bar charts
        col4, col5, col6 = st.columns(3)
        with col4:
            with st.container(border=True):
                st.html(title_with_help_icon(strings.METRICS_NAMES["accuracy"],
                                             strings.HELP_METRICS["accuracy"]))
                st.plotly_chart(render_metric_barchart(metrics, "accuracy", show_title=False), config=config.PLOTLY_CONFIG)
        with col5:
            with st.container(border=True):
                st.html(title_with_help_icon(strings.METRICS_NAMES["precision_weighted"],
                                             strings.HELP_METRICS["precision_weighted"]))
                st.plotly_chart(render_metric_barchart(metrics, "precision_weighted", show_title=False), config=config.PLOTLY_CONFIG)
        with col6:
            with st.container(border=True):
                st.html(title_with_help_icon(strings.METRICS_NAMES["recall_weighted"],
                                             strings.HELP_METRICS["recall_weighted"]))
                st.plotly_chart(
                    render_metric_barchart(metrics, "recall_weighted", show_title=False), config=config.PLOTLY_CONFIG)

    with tab3:
        # Instance consensus overview
        with st.container(border=True):
            st.html(title_with_help_icon(strings.INSTANCE_CONCENSUS_TITLE, strings.HELP_INSTANCES))

            instance_metrics = dataset_obj.get_instance_consensus()
            params = st.query_params.to_dict()
            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            col7, col8, col9, col10 = st.columns(4)

            col7.html(render_metric_card(strings.INSTANCES_TOTAL_TITLE,
                                         instance_metrics['total'],
                                         f"{settings.INSTANCES_FILENAME}?{query_string}&instances=all"))
            col8.html(render_metric_card(strings.INSTANCES_ALL_CORRECT_TITLE,
                                         instance_metrics['all_correct'],
                                         f"{settings.INSTANCES_FILENAME}?{query_string}&instances=allcorrect"))
            col9.html(render_metric_card(strings.INSTANCES_ALL_INCORRECT_TITLE,
                                         instance_metrics['all_incorrect'],
                                         f"{settings.INSTANCES_FILENAME}?{query_string}&instances=allincorrect"))
            col10.html(render_metric_card(strings.INSTANCES_DISAGREE_TITLE,
                                         instance_metrics['models_disagree'],
                                         f"{settings.INSTANCES_FILENAME}?{query_string}&instances=disagree"))

            render_upset_figure(dataset_obj)

if __name__ == "__main__":
    main()