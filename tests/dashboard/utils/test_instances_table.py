"""
Tests for dashboard/utils/instances_table.py

Covers:

- build_display_df
"""
import pytest
import pandas as pd

from xailens import strings
from xailens.dashboard.utils.instances_table import build_display_df

@pytest.fixture
def instances_df():
    """Instance table with one correct and one incorrect prediction per row."""
    return pd.DataFrame({
        strings.DISPLAY_NAMES["instance_uuid"]: ["inst_0", "inst_1"],
        "true_label": [1, 0],
        "Logistic Regression": [1, 1],   # inst_0 correct, inst_1 incorrect
        "XGBoost": [0, 0],               # inst_0 incorrect, inst_1 correct
    })


def test_build_display_df(instances_df):
    """tests that the function puts link on id column, applies styling and adds a view column."""
    result = build_display_df(instances_df, {"dataset": "cleveland"})

    id_col = strings.DISPLAY_NAMES["instance_uuid"]
    view_col = strings.DISPLAY_NAMES["view_detail"]

    # id cell links through to the detail page
    id_cell = result.loc[0, id_col]
    assert 'href="' in id_cell
    assert "dataset=cleveland" in id_cell
    assert "instance_uuid=inst_0" in id_cell

    # correct prediction is green, incorrect is red
    assert "#d4edda" in result.loc[0, "Logistic Regression"]
    assert "#f8d7da" in result.loc[0, "XGBoost"]

    # view column added, original frame untouched
    assert view_col in result.columns
    assert view_col not in instances_df.columns