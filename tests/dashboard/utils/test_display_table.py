"""
Test the functions in xailens.dashboard.utils.display_table

Covers:

- _format_cell
- generate_rows_html
- render_dataframe
- render_series

"""

import pytest
import pandas as pd
import numpy as np
from xailens.dashboard.utils.display_table import (
    _format_cell, generate_table_rows_html, dataframe_to_html_table,
    series_to_html_table)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "instance_uuid": ["a1b2", "c3d4", "e5f6", "g7h8"],
        "age": [63, 45, np.nan, 58],
        "prediction": [1, 0, 1, np.nan],
        "actual": [1, 0, 0, 1],
        "confidence": [0.92, 0.71, np.nan, 0.55],
        "correct": [True, True, False, np.nan],
        "notes": ["typical angina", None, "asymptomatic", pd.NA],
    })

@pytest.fixture
def single_value_series():
    return pd.Series({"age": 63})

@pytest.fixture
def multi_value_series():
    return pd.Series({
        "age": 63,
        "sex": "male",
        "cholesterol": 233.0,
        "confidence": np.nan,
        "notes": None,
    })

@pytest.mark.parametrize("val, expected", [
    (0, "0"),
    (1, "1"),
    (True, "True"),
    (False, "False"),
    (2.120, "2.12"),
    (2.12345, "2.123"),
    (2.2, "2.2"),
    (0.12, "0.12"),
    (0.12345, "0.123"),
    (0.10000, "0.1"),
    ("string", "string"),
    (float('nan'), ""),
])
def test_display_bool_value_non_float(val, expected):
    """ Tests expected formatting of values """
    assert _format_cell(val) == expected

def test_generate_rows_html(sample_df):
    """ Tests generating HTML table """
    html = generate_table_rows_html(sample_df)
    assert "<tr>" in html
    assert sample_df.shape[0] == html.count("<tr>")

def test_render_dataframe(sample_df):
    """Tests creating an HTML table from dataframe """
    html = dataframe_to_html_table(sample_df)
    assert "<table" in html
    assert "report-table" in html

def test_render_series_single(single_value_series):
    """ Tests creating an HTML table from single value series """
    html = series_to_html_table(single_value_series)
    assert "<strong>Age</strong>" in html
    assert "<table" not in html

def test_render_series_multi(multi_value_series):
    """Tests creating an HTML table from multi value series """
    html = series_to_html_table(multi_value_series)
    assert "<table" in html
    # should be length of series plus the header
    assert html.count("<tr>") == len(multi_value_series) + 1


