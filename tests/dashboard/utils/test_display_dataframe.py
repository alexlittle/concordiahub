"""
Tests for display_dataframe

Covers:
- _display_value
- generate_rows_html
- render_dataframe.
"""

import pandas as pd
import pytest

from xailens.dashboard.utils.display_dataframe import (
    generate_instance_rows_html, render_dataframe)
from xailens import strings
from xailens.dashboard import settings


@pytest.fixture(autouse=True)
def patch_lookups(monkeypatch):
    monkeypatch.setattr(settings, "INSTANCE_DETAIL_FILENAME", "instance_detail.html")
    monkeypatch.setattr(strings, "DISPLAY_NAMES", {"instance_uuid": "instance_uuid"})


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "instance_uuid": ["id-1", "id-2"],
        "true_label": [1, 0],
        "model_a": [1, 1],   # row1 correct, row2 incorrect
    })

# -------------------------
# test generate_rows_html
# -----------------------
def test_generate_rows_html_marks_correct_and_incorrect_predictions(sample_df):
    html = generate_instance_rows_html(sample_df, query_params={})

    assert 'class="pred-correct"' in html
    assert 'class="pred-incorrect"' in html
    # row 1: model_a == true_label (1 == 1) -> correct
    # row 2: model_a != true_label (1 != 0) -> incorrect
    rows = html.split("\n")
    assert 'pred-correct">True' in rows[0]
    assert 'pred-incorrect">True' in rows[1]


def test_generate_rows_html_builds_id_link_with_query_params(sample_df):
    html = generate_instance_rows_html(sample_df, query_params={"dataset": "cleveland"})

    assert '<a href="instance_detail.html?dataset=cleveland&instance_uuid=id-1">id-1</a>' in html
    assert '<a href="instance_detail.html?dataset=cleveland&instance_uuid=id-2">id-2</a>' in html


def test_generate_rows_html_true_label_col_renders_plain(sample_df):
    row1 = generate_instance_rows_html(sample_df, query_params={}).split("\n")[0]
    assert "<td>True</td>" in row1


def test_generate_rows_html_skip_cols_excluded_from_styling(sample_df):
    html = generate_instance_rows_html(sample_df, query_params={}, skip_cols=["model_a"])

    assert "pred-correct" not in html
    assert "pred-incorrect" not in html
    assert "<td>True</td>" in html


def test_generate_rows_html_custom_id_col(sample_df):
    df = sample_df.rename(columns={"instance_uuid": "patient_id"})
    html = generate_instance_rows_html(df, query_params={}, id_col="patient_id")

    assert '<a href="instance_detail.html?&instance_uuid=id-1">id-1</a>' in html


def test_generate_rows_html_empty_dataframe_returns_empty_string():
    df = pd.DataFrame(columns=["instance_uuid", "true_label", "model_a"])
    html = generate_instance_rows_html(df, query_params={})
    assert html == ""


# ----------------------
# test render_instances_df
# ----------------------

def test_render_instances_df_includes_all_column_headers(sample_df):
    html = render_dataframe(sample_df, query_params={})
    for col in sample_df.columns:
        assert f"<th nowrap>{col}</th>" in html


def test_render_instances_df_wraps_rows_in_table_body(sample_df):
    html = render_dataframe(sample_df, query_params={})
    assert '<table class="report-table">' in html
    assert "<tbody>" in html
    assert "id-1" in html
    assert "id-2" in html