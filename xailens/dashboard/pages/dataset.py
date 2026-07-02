import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.explorer import get_explorer
from xailens.dashboard.utils.common import display_breadcrumbs
from xailens.dashboard.utils.dataset_helpers import render_metric_barchart, render_metric_card
from xailens.dashboard.utils.heatmap import render_features_heatmap
from xailens.entities.dataset import Dataset
from xailens import strings
from xailens.dashboard import settings

def main():
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

    # show breadcrumb
    st.html(display_breadcrumbs(dataset))

    # Get the selected models for the dataset
    selected_models = explorer.get_selected_models_for_dataset(dataset, selected_model_names)
    dataset_obj = Dataset(dataset, selected_models)

    with st.container(border=True):
        recommended_models = dataset_obj.get_recommended_models()

        if len(recommended_models) == 1:
            st.html(f"<h3>{strings.RECOMMENDED_MODEL_TITLE}: {recommended_models[0]}</h3>")
        else:
            st.html(f"<h3>{strings.RECOMMENDED_MODELS_TITLE}: {', '.join(recommended_models)}</h3>")

    # Get performance metrics
    metrics = dataset_obj.get_metrics()
    metrics = metrics.sort_values(by=['model'])

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        # F1 weighted score
        with st.container(border=True):
            st.html(f"<h3>{strings.PERF_F1_WEIGHTED_TITLE}</h3>")
            st.plotly_chart(render_metric_barchart(metrics, "f1_weighted", show_title=False))
    with col2:
        # Global explanations
        with st.container(border=True):
            st.html(f"<h3>{strings.GLOBAL_EXPLANATION_TITLE}</h3>")
            df_global_exp = dataset_obj.get_global_explanations()
            st.plotly_chart(render_features_heatmap(df_global_exp), width='content')
    with col3:
        # Instance consensus overview
        with st.container(border=True):
            st.html(f"<h3>{strings.INSTANCE_CONCENSUS_TITLE}</h3>")

            instance_metrics = dataset_obj.get_instance_consensus()
            params = st.query_params.to_dict()
            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            st.html(render_metric_card(strings.INSTANCES_TOTAL_TITLE,
                                         instance_metrics['total'],
                                         f"{settings.INSTANCES_FILENAME}?{query_string}&instances=all"))
            st.html(render_metric_card(strings.INSTANCES_ALL_CORRECT_TITLE,
                                         instance_metrics['all_correct'],
                                         f"{settings.INSTANCES_FILENAME}?{query_string}&instances=allcorrect"))
            st.html(render_metric_card(strings.INSTANCES_ALL_INCORRECT_TITLE,
                                         instance_metrics['all_incorrect'],
                                         f"{settings.INSTANCES_FILENAME}?{query_string}&instances=allincorrect"))
            st.html(render_metric_card(strings.INSTANCES_DISAGREE_TITLE,
                                         instance_metrics['models_disagree'],
                                         f"{settings.INSTANCES_FILENAME}?{query_string}&instances=disagree"))


    # accuracy/precision/recall bar charts
    with st.expander(f"{strings.PERF_METRICS_TITLE}", expanded=False):
        col4, col5, col6 = st.columns(3)
        with col4:
            with st.container(border=True):
                st.plotly_chart(render_metric_barchart(metrics, "accuracy"))
        with col5:
            with st.container(border=True):
                st.plotly_chart(render_metric_barchart(metrics, "precision_weighted"))
        with col6:
            with st.container(border=True):
                st.plotly_chart(render_metric_barchart(metrics, "recall_weighted"))


if __name__ == "__main__":
    main()