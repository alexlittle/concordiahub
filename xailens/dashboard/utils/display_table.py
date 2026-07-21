"""
Functions for displaying a pd.Dataframe as an HTML table
"""

from typing import Any

import pandas as pd
import numbers

from markdown import markdown


def _format_cell(value: Any) -> str:
    """
    Helper function to format a cell

    Args:
        value (Any): Value to format

    Returns:
        str: Formatted value
    """
    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        if pd.isna(value):
            return ""
        formatted = f"{value:.3f}".rstrip("0").rstrip(".")
        return formatted
    return str(value)


def generate_table_rows_html(df: pd.DataFrame) -> str:
    """
    Generates HTML table rows for the given dataframe

    Args:
        df (pd.DataFrame): dataframe

    Returns:
        str: HTML code for the table rows
    """
    rows = []
    for idx, row in df.iterrows():
        cells = "".join(f"<td>{_format_cell(row[col])}</td>" for col in df.columns)
        rows.append(f"<tr>{cells}</tr>")
    return "\n".join(rows)

def dataframe_to_html_table(df: pd.DataFrame) -> str:
    """
    Generates an HTML table for the given dataframe

    Args:
        df (pd.DataFrame): dataframe

    Returns:
        str: HTML code for the table
    """
    rows_html = generate_table_rows_html(df)
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


def series_to_html_table(series: pd.Series) -> str:
    """
    Generates a HTML table code for the given series

    Args:
        series (pd.Series): series

    Returns:
        str: HTML code for the table
    """

    if len(series) == 1:
        name, value = next(series.items())
        return f'<p><strong>{name.capitalize()}</strong>:</p><p>{markdown(str(value))}</p>'

    rows = [
        f"<tr><td>{name}</td><td>{markdown(str(value))}</td></tr>" for name, value in series.items()
    ]
    rows_html = "\n".join(rows)
    return f"""
        <table class="report-table">
            <thead>
                <tr>
                    <th nowrap>Feature Name</th>
                    <th nowrap>Value</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """