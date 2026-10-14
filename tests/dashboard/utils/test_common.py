"""
Tests for the common utils

Covers:

- display_breadcrumbs
- title_with_help_icon
- display_bool_value
- rename_model_prediction_columns
"""
import pytest
import pandas as pd

from concordiahub.dashboard.utils.common import (
    display_breadcrumbs, title_with_help_icon, display_bool_value,
    rename_model_prediction_columns)
from concordiahub import strings

#-----------------
# display_breadcrumbs
#-----------------
def test_no_params():
    """Test with no params """
    breadcrumbs = display_breadcrumbs()
    assert breadcrumbs == "<div class='breadcrumbs'><a href='/' target='_self'>Home</a></div>"

def test_dataset():
    """Test with dataset param """
    breadcrumbs = display_breadcrumbs(dataset="cleveland")
    assert breadcrumbs == "<div class='breadcrumbs'><a href='/' target='_self'>Home</a> &nbsp;&gt;&nbsp; cleveland</div>"


def test_invalid_instance_group():
    """Test with invalid instance_group param """
    with pytest.raises(KeyError):
        display_breadcrumbs(instance_group="inst group")

def test_valid_instance_group():
    """Test with invalid instance_group param """
    breadcrumbs = display_breadcrumbs(instance_group="all")
    assert breadcrumbs == "<div class='breadcrumbs'><a href='/' target='_self'>Home</a> &nbsp;&gt;&nbsp; All Instances</div>"

def test_instance_id():
    """Test with instance_id param """
    breadcrumbs = display_breadcrumbs(instance_uuid="uuid_0")
    assert breadcrumbs == "<div class='breadcrumbs'><a href='/' target='_self'>Home</a> &nbsp;&gt;&nbsp; uuid_0</div>"


def test_matrix_set():
    """Test with matrix set param """
    breadcrumbs = display_breadcrumbs(matrix_set="lr")
    assert breadcrumbs == "<div class='breadcrumbs'><a href='/' target='_self'>Home</a> &nbsp;&gt;&nbsp; Misclassifications</div>"


def test_matrix_set_and_instance_id():
    """Test with matrix set and instance id param """
    breadcrumbs = display_breadcrumbs(matrix_set="lr", instance_uuid="uuid_0")
    print(breadcrumbs)
    link = f"<a href='/instances?dataset=None&matrix_set=lr' target='_self'>{strings.MISCLASSIFIED}</a>"

    assert breadcrumbs == f"<div class='breadcrumbs'><a href='/' target='_self'>Home</a> &nbsp;&gt;&nbsp; {link} &nbsp;&gt;&nbsp; uuid_0</div>"

#-----------------
# title_with_help_icon
#-----------------

def test_title_with_help_icon():
    """Tests title with help icon """
    result = title_with_help_icon("FakeTitle","Some fake help text")

    assert "Some fake help text" in result
    assert "FakeTitle" in result
    assert "<img" in result


# -----------------
# test display_bool_value
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
def test_display_bool_value_non_float(val, expected):
    assert display_bool_value(val) == expected


@pytest.mark.parametrize("val, expected", [
    (0.0, "0.0"),
    (1.0, "1.0"),
])
def test_display_bool_value_float_bypasses_bool_conversion(val, expected):
    # Floats that are numerically 0/1 should NOT be rendered as True/False,
    assert display_bool_value(val) == expected

# -----------------
# test rename_model_prediction_columns
# ----------------
def test_rename_model_prediction_columns(monkeypatch):
    """Test with rename model prediction column param """
    monkeypatch.setattr(strings, "DISPLAY_NAMES", {"instance_id": "Instance"})
    df = pd.DataFrame({
        "instance_id": ["inst_0"],
        "logistic_regression": [1],
        "xgboost": [0],
    })

    result = rename_model_prediction_columns(df)

    assert list(result.columns) == [
        "Instance", "logistic_regression Prediction", "xgboost Prediction"]