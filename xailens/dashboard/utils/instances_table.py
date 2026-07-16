"""
For displaying the instances table
"""

import pandas as pd

from typing import Optional

from xailens import strings
from xailens.dashboard import settings
from xailens.dashboard.utils.common import display_bool_value


def build_display_df(df: pd.DataFrame,
                      query_params: dict,
                      true_label_col: Optional[str] = "true_label",
                      id_col: Optional[str] = None,
                      skip_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Builds a iTable specific dataframe with the specific styles needed and adds
    the link to go to the full instance detail page

    Args:
        df (pd.DataFrame): original dataframe to update
        query_params (dict): original query params for adding to the link
        true_label_col (Optional[str]):  column name containing the true label,
            defaults to "true_label"
        id_col (Optional[str]): column name containing the instance id,
            defaults to None
        skip_cols (Optional[list]): list of any other columns that should not be
            displayed, defaults to None

    Returns:
        pd.DataFrame: updated dataframe

    """
    skip_cols = skip_cols or []
    id_display_col = id_col or strings.DISPLAY_NAMES["instance_uuid"]
    pred_cols = [c for c in df.columns if c not in [true_label_col,
                                                    id_display_col] + skip_cols]

    cell_style = ("display:inline-block; "
                  "padding:8px 10px; "
                  "white-space:normal; "
                  "word-break:break-word;")
    correct_style = cell_style + " background-color:#d4edda; color:#155724;"
    incorrect_style = cell_style + " background-color:#f8d7da; color:#721c24;"

    display_df = df.copy()
    query_string = "&".join(f"{k}={v}" for k, v in query_params.items())

    display_df[id_display_col] = df[id_display_col].apply(
        lambda val: (
        f'<a href="{settings.INSTANCE_DETAIL_FILENAME}?{query_string}&instance_uuid={val}" '
        f'style="{cell_style}">{val}</a>')
    )

    for col in pred_cols:
        display_df[col] = df.apply(
            lambda row, c=col: (
                f'<span style="{correct_style if row[c] == row[true_label_col] else incorrect_style}">'
                f'{display_bool_value(row[c])}</span>'),
            axis=1)

    for col in df.columns.difference(pred_cols + [id_display_col]):
        display_df[col] = df[col].apply(
            lambda v:
            f'<span style="{cell_style}">{display_bool_value(v)}</span>')

    return display_df