import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.explorer import get_explorer
from xailens.dashboard.utils.common import display_breadcrumbs
from xailens.dashboard.utils.display_instances_df import render_instances_df
from xailens import strings
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
                 dataset_id=dataset,
                 selected_instance_group=instance_group,)

    # Get the selected models for the dataset and matching instances
    selected_modelevals = explorer.get_selected_modelevals_for_dataset(dataset, selected_model_names)
    dataset_obj = Dataset(dataset, selected_modelevals)
    df_instances = dataset_obj.get_all_instances()
    if instance_group != "all":
        df_instances = df_instances[df_instances['agreement'] == instance_group]

    # breadcrumb
    st.html(display_breadcrumbs(dataset, instance_group=instance_group))

    st.html(f"<h2>{strings.INSTANCE_GROUPS[instance_group]} ({len(df_instances)})</h2>")



    # Display matching instances
    # remove agreement column so doesn't display
    df_instances = df_instances.drop(columns="agreement")
    df_instances = df_instances.rename(columns=strings.DISPLAY_NAMES)
    query_params = dict(st.query_params)
    st.html(render_instances_df(df_instances, query_params, true_label_col=strings.DISPLAY_NAMES["true_label"]))



if __name__ == "__main__":
    main()