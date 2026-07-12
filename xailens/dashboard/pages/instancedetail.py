import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.explorer import get_explorer
from xailens import strings, config
from xailens.dashboard.utils.common import display_breadcrumbs
from xailens.dashboard.utils.instancedetail_helpers import build_model_panel, render_model_panel
from xailens.dashboard.utils.display_table import render_series
from xailens.dashboard.utils.display_instances_df import render_instances_df
from xailens.entities.dataset import Dataset
from xailens.dashboard.utils.heatmap import render_features_heatmap

def main():
    explorer = get_explorer()

    dataset = st.query_params.get('dataset')
    models_param = st.query_params.get("models")
    instance_group = st.query_params.get("instances")
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

    st.html(display_breadcrumbs(dataset, instance_group=instance_group, instance_uuid=instance_uuid))


    selected_models = explorer.get_selected_models_for_dataset(dataset, selected_model_names)
    dataset_obj = Dataset(dataset, selected_models)

    # Overall predictions
    st.html(f"<h2>{strings.INSTANCE_DETAIL_PREDICTIONS_TITLE}</h2>")
    df_instances = dataset_obj.get_all_instances()
    df_instance = df_instances[df_instances["instance_uuid"] == instance_uuid]

    df_instance = df_instance.drop(columns=["instance_uuid", "agreement"])

    rename_map = {
        col: strings.DISPLAY_NAMES.get(col, f"{col} Prediction")
        for col in df_instance.columns
    }

    df_instance = df_instance.rename(columns=rename_map)

    st.html(render_instances_df(df_instance, true_label_col=strings.DISPLAY_NAMES["true_label"]))

    col1, col2 = st.columns(2, gap="large")

    with col1:
    # display the local feature heatmap
        st.html(f"<h2>{strings.INSTANCE_DETAIL_HEATMAP_TITLE}</h2>")
        df_local_exp = dataset_obj.get_local_explanations(instance_uuid)
        st.plotly_chart(render_features_heatmap(df_local_exp), width='content', config=config.PLOTLY_CONFIG)

    with col2:
        st.html(f"<h2>{strings.INSTANCE_DETAIL_DATA_TITLE}</h2>")

        # display the original data
        instance_data = dataset_obj.get_instance_data(instance_uuid)

        st.html(render_series(instance_data))

    # Panels for each model explanation
    st.html(f"<h2>{strings.INSTANCE_DETAIL_MODEL_EXPLANATIONS}</h2>")
    for i in range(0, len(selected_models), 2):
        cols = st.columns(2, gap="large")
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(selected_models):
                me = selected_models[idx]
                with col:
                    key = me.get("model_uuid")
                    viz_state_key = f"model-viz-{key}"
                    panel = build_model_panel(me, viz_state_key, st.session_state, instance_uuid)
                    render_model_panel(panel, f"explanation_{key}")

if __name__ == "__main__":
    main()