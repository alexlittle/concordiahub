import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.explorer import get_explorer
from xailens import strings
from xailens.dashboard.utils.common import display_breadcrumbs
from xailens.dashboard.utils.instancedetail_helpers import build_model_panel, render_model_panel
from xailens.dashboard.utils.display_table import render_series
from xailens.entities.dataset import Dataset
from xailens.dashboard.utils.heatmap import render_features_heatmap

def main():
    explorer = get_explorer()

    dataset = st.query_params.get('dataset')
    models_param = st.query_params.get("models")
    instance_group = st.query_params.get("instances")
    instance_uuid = st.query_params.get("instance_uuid")

    all_modelevals = explorer.get_modelevals_for_dataset(dataset)
    all_model_names = [me.get("model_display_name") for me in all_modelevals]

    if models_param:
        selected_model_names = sorted(models_param.split(","))
    else:
        selected_model_names = sorted(all_model_names)

    apply_layout(strings.INSTANCE_DETAIL_TITLE.format(dataset=dataset),
                 all_model_names=all_model_names,
                 selected_model_names=selected_model_names,
                 dataset_id=dataset,)

    st.html(display_breadcrumbs(dataset, instance_group=instance_group, instance_uuid=instance_uuid))


    selected_modelevals = explorer.get_selected_modelevals_for_dataset(dataset, selected_model_names)
    dataset_obj = Dataset(dataset, selected_modelevals)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        # display the local feature heatmap
        st.html(f"<h2>{strings.INSTANCE_DETAIL_HEATMAP_TITLE}</h2>")
        df_local_exp = dataset_obj.get_local_explanations(instance_uuid)
        st.plotly_chart(render_features_heatmap(df_local_exp), width='content')

    with col2:
        st.html(f"<h2>{strings.INSTANCE_DETAIL_DATA_TITLE}</h2>")

        # display the original data
        x_test, ground_truth = dataset_obj.get_instance_data(instance_uuid)

        st.html(f"<strong>True Value</strong>: {ground_truth}")
        st.html(render_series(x_test))

    # Panels for each model explanation
    for i in range(0, len(selected_modelevals), 2):
        cols = st.columns(2, gap="large")
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(selected_modelevals):
                me = selected_modelevals[idx]
                with col:
                    key = me.get("model_uuid")
                    panel = build_model_panel(me, f"model-viz-{key}", st.query_params)
                    render_model_panel(panel, st.query_params, plot_key=f"explanation_{key}")

if __name__ == "__main__":
    main()