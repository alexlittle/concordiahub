import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.services import get_explorer
from xailens.dashboard.utils.render.common import display_breadcrumbs
from xailens import strings, constants
from xailens.entities.dataset import Dataset


def main():
    explorer = get_explorer()

    dataset = st.query_params.get('dataset')
    models_param = st.query_params.get("models")
    instance_group = st.query_params.get("instances")

    all_modelevals = explorer.get_modelevals_for_dataset(dataset)
    all_model_names = [me.get("model_display_name") for me in all_modelevals]

    if models_param:
        selected_model_names = sorted(models_param.split(","))
    else:
        selected_model_names = sorted(all_model_names)

    apply_layout(strings.INSTANCE_TITLE.format(dataset=dataset),
                 all_model_names=all_model_names,
                 selected_model_names=selected_model_names,
                 dataset_id=dataset)

    # breadcrumb
    st.html(display_breadcrumbs(dataset, instance_group=instance_group))

    st.html(f"<h2>{constants.INSTANCE_GROUPS[instance_group]}</h2>")
    # get matching instances
    # Get the selected models for the dataset
    selected_modelevals = explorer.get_selected_modelevals_for_dataset(dataset, selected_model_names)
    dataset_obj = Dataset(dataset, selected_modelevals)
    df_instances = dataset_obj.get_all_instances()
    if instance_group != "all":
        df_instances = df_instances[df_instances['agreement']==instance_group]

    st.dataframe(df_instances)

    # display matching instances


if __name__ == "__main__":
    main()