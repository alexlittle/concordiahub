"""
Helper functions for common dashboard display elements
"""
import pandas as pd
import streamlit as st

from typing import Optional, Any

from xailens.dashboard import settings
from xailens import strings
from xailens.dashboard.assets.icons import help_icon


def _breadcrumb_instance_group_set(
        dataset: Optional[str],
        model_param: str,
        instance_group: Optional[str] = None,
        instance_uuid: Optional[str] = None,
        matrix_set: Optional[str] = None) -> list[str]:
    """
    Generates correct breadcrumbs based on whether an instance group or matrix
    set is provided

    Args:
        dataset (Optional[str]): dataset name
        model_param (str): string for url query param for the models selected
        instance_group (Optional[str]): instance group name
        instance_uuid (Optional[str]):  instance uuid
        matrix_set (Optional[str]):  matrix sets (as comma separated string)

    Returns:
        list[str]: list of items to include as breadcrumbs
    """

    parts = []
    if instance_group and not matrix_set:
        if instance_uuid:
            parts.append(f"""
<a href='/{settings.INSTANCES_FILENAME}?dataset={dataset}{model_param}&instances={instance_group}' target='_self'>
{strings.INSTANCE_GROUPS[instance_group]}</a>""")
        else:
            parts.append(strings.INSTANCE_GROUPS[instance_group])

    if matrix_set and not instance_group:
        if instance_uuid:
            parts.append(f"<a href='/{settings.INSTANCES_FILENAME}?dataset={dataset}{model_param}&matrix_set={matrix_set}' target='_self'>{strings.MISCLASSIFIED}</a>")
        else:
            parts.append(strings.MISCLASSIFIED)
    return parts


def display_breadcrumbs(dataset: Optional[str] = None,
                        instance_group: Optional[str] = None,
                        instance_uuid: Optional[str] = None,
                        matrix_set: Optional[str] = None) -> str:
    """
    Display breadcrumbs for a page. Works out what the breadcrumbs are based on
    which params are set.

    Args:
        dataset (Optional[str]): dataset name
        instance_group (Optional[str]): instance group name
        instance_uuid (Optional[str]):  instance uuid
        matrix_set (Optional[str]):  matrix sets (as comma separated string)

    Returns:
        str: HTML code to display breadcrumbs
    """
    parts = ["<a href='/' target='_self'>Home</a>"]

    models = st.query_params.get('models')
    model_param = f"&models={models}" if models else ""

    ds_tab = st.query_params.get('ds_tab')
    tab_param = f"&ds_tab={ds_tab}" if ds_tab else ""

    if dataset:
        if instance_group or matrix_set:
            parts.append(f"""
                <a href='/{settings.DATASET_FILENAME}?dataset={dataset}{model_param}{tab_param}'
                target='_self'>{dataset}</a>""")
        else:
            parts.append(f"{dataset}")

    parts.extend(_breadcrumb_instance_group_set(
        dataset, model_param, instance_group, instance_uuid, matrix_set))

    if instance_uuid:
        parts.append(f"{instance_uuid}")
    
    return f"<div class='breadcrumbs'>{' &nbsp;&gt;&nbsp; '.join(parts)}</div>"


def title_with_help_icon(title: str, help_text:str) -> str:
    """
    Display a title with help icon/text

    Args:
        title (str): title
        help_text (str): help text

    Returns:
        str: HTML code to display title and help icon/text
    """
    return f"""
            <div class="card-header-row">
                <h3>{title}</h3>
                {help_icon(help_text)}
            </div>
        """

def display_bool_value(val: Any) -> str:
    """
    Function to convert bool value to string of True or False

    Args:
        val (Any): value to be converted

    Returns:
        str: string value of the val provided
    """
    if val in (0, 1, True, False) and not isinstance(val, float):
        return str(bool(val))
    return str(val)

def rename_model_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames a model name column to be model name + "prediction"

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: dataframe with renamed columns
    """

    rename_map = {
        col: strings.DISPLAY_NAMES.get(col, f"{col} Prediction")
        for col in df.columns
    }
    df = df.rename(columns=rename_map)
    return df