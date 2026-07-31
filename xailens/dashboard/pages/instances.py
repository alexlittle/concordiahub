"""
Streamlit page to display instances and misclassifications
"""

import streamlit as st
from itables.streamlit import interactive_table

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.explorer import get_explorer
from xailens.dashboard.utils.common import (
    display_breadcrumbs, rename_model_prediction_columns)
from xailens.dashboard.utils.instances_table import build_display_df
from xailens import strings
from xailens.entities.dataset import Dataset
from xailens.dashboard.utils.matrix_set import render_matrixset_figure
from xailens.dashboard import settings

def main():
    """
    Streamlit page to display instances and misclassifications
    """

    explorer = get_explorer()

    # get the query params
    dataset = st.query_params.get('dataset')
    models_param = st.query_params.get("models")
    instance_group = st.query_params.get("instances", None)

    all_models = explorer.get_models_for_dataset(dataset)
    all_model_names = [me.get("model_display_name") for me in all_models]

    if models_param:
        selected_model_names = sorted(models_param.split(","))
    else:
        selected_model_names = sorted(all_model_names)

    apply_layout(strings.INSTANCE_TITLE.format(dataset=dataset),
                 all_model_names=all_model_names,
                 selected_model_names=selected_model_names,
                 dataset_id=dataset,
                 selected_instance_group=instance_group)

    # Get the selected models for the dataset and matching instances
    selected_models = explorer.get_selected_models_for_dataset(
        dataset, selected_model_names)
    dataset_obj = Dataset(dataset, selected_models)
    df_instances = dataset_obj.get_all_instances()

    breadcrumb_slot = st.empty()
    heading_slot = st.empty()

    # only show the matrix plot if a matrix set if given (i.e. there isn't
    # an instance group
    if instance_group is None:
        render_matrixset_figure(dataset_obj, navigate_on_click=False)

    matrix_set = st.query_params.get("matrix_set", None)

    # check that only one of matrix_set or instance_group is present
    if (matrix_set is not None and instance_group is not None) or \
            (matrix_set is None and instance_group is None):
        st.warning(strings.WARNING_INVALID_URL_PARAMS)
        st.stop()


    # determine which instances to show based on if it's an instance_group or a
    # matrix_set
    if instance_group is not None and instance_group != "all":
        df_instances = df_instances[df_instances['agreement'] == instance_group]

    if instance_group is None and matrix_set is not None:
        df_instances = dataset_obj.get_instances_for_set(matrix_set)

    # breadcrumb
    breadcrumb_slot.html(
        display_breadcrumbs(dataset,
                            instance_group=instance_group,
                            matrix_set=matrix_set))

    # page title
    if instance_group is not None:
        heading_slot.html(
            f"""<h2>{strings.INSTANCE_GROUPS[instance_group]}
            ({len(df_instances)})</h2>""")
    else:
        heading_slot.html(f"<h2>{strings.MISCLASSIFIED}</h2>")

    # Display matching instances
    # remove agreement column so doesn't display
    df_instances = df_instances.drop(columns="agreement")

    # append "prediction" to the column title
    df_instances = rename_model_prediction_columns(df_instances)

    # display the table of instances
    query_params = dict(st.query_params)

    if matrix_set is not None:
        # put in alpha order
        models = sorted(matrix_set.split(","))
        model_set = ", ".join(models)
        st.html(f"<h3>"
                f"{strings.MISCLASSIFIED_INSTANCES_TITLE.format(
                    models=model_set)}</h3>")

    interactive_table(
        build_display_df(df_instances,
                         query_params,
                         strings.DISPLAY_NAMES["true_label"]),
        allow_html=True,
        # maxBytes/maxRows need to be set to make sure it doesn't max out at 64k
        maxBytes=settings.ITABLE_MAX_BYTES,
        maxRows=settings.ITABLE_MAX_ROWS,
        pageLength=settings.ITABLE_ROWS_PER_PAGE,
        showIndex=False, # don't display the id column
        classes="display compact",
    )



if __name__ == "__main__":
    main()