"""
Tests for the common utils

Covers:

- display_breadcrumbs
- title_with_help_icon
- display_bool_value
- rename_model_prediction_columns
"""
import pytest

from xailens.dashboard.utils.common import display_breadcrumbs
from xailens import strings

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