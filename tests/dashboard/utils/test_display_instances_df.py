"""
Tests for _display_value, generate_rows_html, and render_instances_df.
"""
import pandas as pd
import pytest

from xailens.dashboard.utils.display_instances_df import _display_value, generate_rows_html, render_instances_df
from xailens import strings
from xailens.dashboard import settings



# -----------------
# test  _display_value
# ----------------

@pytest.mark.parametrize("val, expected", [
    (0, "False"),
    (1, "True"),
    (True, "True"),
    (False, "False"),
    (2, "2"),
    ("abc", "abc"),
    (None, "None"),
])
def test_display_value_non_float(val, expected):
    assert _display_value(val) == expected


@pytest.mark.parametrize("val, expected", [
    (0.0, "0.0"),
    (1.0, "1.0"),
])
def test_display_value_float_bypasses_bool_conversion(val, expected):
    # Floats that are numerically 0/1 should NOT be rendered as True/False,
    # since the isinstance(val, float) check excludes them.
    assert _display_value(val) == expected


# -------------------------
# test generate_rows_html
# -----------------------

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


def test_generate_rows_html_marks_correct_and_incorrect_predictions(sample_df):
    html = generate_rows_html(sample_df, query_params={})

    assert 'class="pred-correct"' in html
    assert 'class="pred-incorrect"' in html
    # row 1: model_a == true_label (1 == 1) -> correct
    # row 2: model_a != true_label (1 != 0) -> incorrect
    rows = html.split("\n")
    assert 'pred-correct">True' in rows[0]
    assert 'pred-incorrect">True' in rows[1]


def test_generate_rows_html_builds_id_link_with_query_params(sample_df):
    html = generate_rows_html(sample_df, query_params={"dataset": "cleveland"})

    assert '<a href="instance_detail.html?dataset=cleveland&instance_uuid=id-1">id-1</a>' in html
    assert '<a href="instance_detail.html?dataset=cleveland&instance_uuid=id-2">id-2</a>' in html


def test_generate_rows_html_true_label_col_renders_plain(sample_df):
    row1 = generate_rows_html(sample_df, query_params={}).split("\n")[0]
    assert "<td>True</td>" in row1


def test_generate_rows_html_skip_cols_excluded_from_pred_styling(sample_df):
    html = generate_rows_html(sample_df, query_params={}, skip_cols=["model_a"])

    assert "pred-correct" not in html
    assert "pred-incorrect" not in html
    assert "<td>True</td>" in html


def test_generate_rows_html_custom_id_col(sample_df):
    df = sample_df.rename(columns={"instance_uuid": "patient_id"})
    html = generate_rows_html(df, query_params={}, id_col="patient_id")

    assert '<a href="instance_detail.html?&instance_uuid=id-1">id-1</a>' in html


def test_generate_rows_html_empty_dataframe_returns_empty_string():
    df = pd.DataFrame(columns=["instance_uuid", "true_label", "model_a"])
    html = generate_rows_html(df, query_params={})
    assert html == ""


# ---------------------------------------------------------------------------
# test render_instances_df
# ---------------------------------------------------------------------------

def test_render_instances_df_includes_all_column_headers(sample_df):
    html = render_instances_df(sample_df, query_params={})
    for col in sample_df.columns:
        assert f"<th nowrap>{col}</th>" in html


def test_render_instances_df_wraps_rows_in_table_body(sample_df):
    html = render_instances_df(sample_df, query_params={})
    assert '<table class="report-table">' in html
    assert "<tbody>" in html
    assert "id-1" in html
    assert "id-2" in html