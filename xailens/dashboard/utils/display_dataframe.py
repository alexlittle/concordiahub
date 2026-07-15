"""
Helper functions for displaying dataframes as HTML tables on the dashboard
"""
from typing import Optional

import pandas as pd

from xailens import strings
from xailens.dashboard import settings
from xailens.dashboard.utils.common import display_bool_value


def generate_rows_html(df: pd.DataFrame,
                       query_params: dict,
                       true_label_col: Optional[str] = "true_label",
                       id_col: Optional[str] = None,
                       skip_cols: Optional[list] = None) -> str:
    """
    Generates the rows for an HTML table of the instances, including the urls

    Args:
        df (pd.DataFrame): dataframe containing instances data
        query_params (dict): query parameters to append to any links
        true_label_col (Optional[str]): column name containing true label
        id_col (Optional[str]): instance id column name
        skip_cols (Optional[list]): list of any columns that should not be
            displayed

    Returns:
        str: HTML code for table rows
    """
    skip_cols = skip_cols or []
    id_display_col = id_col or strings.DISPLAY_NAMES["instance_uuid"]
    pred_cols = [c for c in df.columns if c not in [true_label_col, id_display_col] + skip_cols]
    rows = []
    for idx, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if col == id_display_col:
                query_string = "&".join(f"{k}={v}" for k, v in query_params.items())
                cells.append(f'<td><a href="{settings.INSTANCE_DETAIL_FILENAME}?{query_string}&instance_uuid={val}">{val}</a></td>')
            elif col in pred_cols:
                css_class = "pred-correct" if val == row[true_label_col] else "pred-incorrect"
                cells.append(f'<td class="{css_class}">{display_bool_value(val)}</td>')
            else:
                cells.append(f"<td>{display_bool_value(val)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return "\n".join(rows)

def render_dataframe(df:pd.DataFrame,
                        query_params:dict = None,
                        true_label_col:str = "true_label",
                        id_col:str = None,
                        skip_cols:str = None)-> str:
        """
        Generates a full HTML table of the instances, including the urls

        Args:
            df (pd.DataFrame): dataframe containing instances data
            query_params (dict): query parameters to append to any links
            true_label_col (str): column name containing true label
            id_col (str): instance id column name
            skip_cols (str ): other columns that should not be displayed

        Returns:
            str: of HTML table rows
        """
        rows_html = generate_rows_html(
            df, query_params, true_label_col, id_col, skip_cols)
        header = "".join(f"<th nowrap>{col}</th>" for col in df.columns)
        return f"""
        <table class="report-table">
            <thead>
                <tr>
                    {header}
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """
