"""
Streamlit page to display details of a specific instance
"""

import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.explorer import get_explorer
from xailens import strings, config
from xailens.dashboard.utils.common import (
    display_breadcrumbs, rename_model_prediction_columns)
from xailens.dashboard.utils.instancedetail_helpers import (
    build_model_panel, render_model_panel)
from xailens.dashboard.utils.display_table import series_to_html_table
from xailens.dashboard.utils.display_dataframe import render_dataframe
from xailens.entities.dataset import Dataset
from xailens.dashboard.utils.heatmap import render_features_heatmap

def main():
    """
    Streamlit page to display details of a specific instance
    """

    explorer = get_explorer()

    # get all the query params
    dataset = st.query_params.get('dataset')
    models_param = st.query_params.get("models")
    instance_group = st.query_params.get("instances")
    matrix_set = st.query_params.get("matrix_set")
    instance_uuid = st.query_params.get("instance_uuid")

    all_models = explorer.get_models_for_dataset(dataset)
    all_model_names = [model.get("model_display_name") for model in all_models]

    if models_param:
        selected_model_names = sorted(models_param.split(","))
    else:
        selected_model_names = sorted(all_model_names)

    apply_layout(strings.INSTANCE_DETAIL_TITLE.format(dataset=dataset),
                 all_model_names=all_model_names,
                 selected_model_names=selected_model_names,
                 dataset_id=dataset,)

    st.html(display_breadcrumbs(dataset,
                                instance_group=instance_group,
                                matrix_set=matrix_set,
                                instance_uuid=instance_uuid))


    selected_models = explorer.get_selected_models_for_dataset(
        dataset, selected_model_names)
    dataset_obj = Dataset(dataset, selected_models)

    tab1, tab2 = st.tabs(strings.INSTANCE_DETAIL_TABS)

    with tab1:
        # Overall predictions
        st.html(f"<h3>{strings.INSTANCE_DETAIL_PREDICTIONS_TITLE}</h3>")
        df_instances = dataset_obj.get_all_instances()
        df_instance = df_instances[df_instances["instance_uuid"] == instance_uuid]
        df_instance = df_instance.drop(columns=["instance_uuid", "agreement"])

        # append "prediction" to the column title
        df_instance = rename_model_prediction_columns(df_instance)

        # show table of the predictions for each model
        st.html(render_dataframe(
            df_instance, true_label_col=strings.DISPLAY_NAMES["true_label"]))

        col1, col2 = st.columns(2, gap="large")

        with col1:
            # display the local feature heatmap
            st.html(f"<h3>{strings.INSTANCE_DETAIL_HEATMAP_TITLE}</h3>")
            df_local_exp = dataset_obj.get_local_explanations(instance_uuid)
            st.plotly_chart(
                render_features_heatmap(df_local_exp),
                width='content',
                config=config.PLOTLY_CONFIG)

        with col2:
            # display the original data
            st.html(f"<h3>{strings.INSTANCE_DETAIL_DATA_TITLE}</h3>")
            instance_data = dataset_obj.get_instance_data(instance_uuid)
            st.html(series_to_html_table(instance_data))

    with tab2:
        # Panels for each model explanation - split so there are 2 columns
        for i in range(0, len(selected_models), 2):
            cols = st.columns(2, gap="large")
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(selected_models):
                    model = selected_models[idx]
                    with col:
                        key = model.get("model_uuid")
                        viz_state_key = f"model-viz-{key}"
                        panel = build_model_panel(model,
                                                  viz_state_key,
                                                  st.session_state,
                                                  instance_uuid)
                        render_model_panel(panel, f"explanation_{key}")

if __name__ == "__main__":
    main()