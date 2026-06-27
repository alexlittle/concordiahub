import streamlit as st
import plotly
from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.explorer import get_explorer
from xailens import strings, constants
from xailens.dashboard.utils.common import display_breadcrumbs
from xailens.dashboard.utils.instancedetail_helpers import build_model_panel, render_model_panel

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

    # display the local feature heatmap

    # display the original data

    # Panels for each model explanation
    for me in selected_modelevals:
        key = me.get("model_uuid")
        panel = build_model_panel(me, f"model-viz-{key}", st.query_params)
        render_model_panel(panel, st.query_params, plot_key=f"explanation_{key}")




if __name__ == "__main__":
    main()