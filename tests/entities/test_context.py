"""
Tests for context.py dataclasses
Since all dataclasses, there's not so much that it is available to test - essentially
only the ordering of the params, so these tests only cover the ordering
"""

from unittest.mock import MagicMock

from xailens.entities.context import ModelContext, DataSchema


def test_model_context_field_order():
    """Tests param positions are as expected"""
    data = MagicMock()
    ctx = ModelContext("logistic_regression", ["coefficients"], "cleveland",
                       data, "Logistic Regression")

    assert ctx.model_type == "logistic_regression"
    assert ctx.xai_methods == ["coefficients"]
    assert ctx.dataset_id == "cleveland"
    assert ctx.data is data
    assert ctx.model_display_name == "Logistic Regression"
    assert ctx.model is None
    assert ctx.model_subtype is None
    assert ctx.model_dir_name is None
    assert ctx.response_schema is None


def test_data_schema_order():
    """Tests param positions are as expected"""
    schema = DataSchema("instance_uuid", "target", "prediction")
    assert schema.instance_id_column == "instance_uuid"
    assert schema.target_column == "target"
    assert schema.prediction_column == "prediction"
    assert schema.feature_columns is None
    assert schema.class_names is None