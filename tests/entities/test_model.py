"""
Tests for Model.load_targets.

Assumes tests/fixtures/model_fixtures.py fixtures (lr_model, xgboost_model,
llm_model, lr_model_factory, etc.) are wired in via conftest.py.
"""
import pandas as pd
import pytest

import xailens.config as config


# -----------------------
# Test get function
# ---------------------

def test_get_returns_nested_value(lr_model):
    assert lr_model.get(config.DATA_SCHEMA_TARGET_COLUMN) == "target"


def test_get_returns_nested_value_via_full_dotted_path(lr_model):
    assert lr_model.get("data_schema.instance_id_column") == "instance_id"


def test_get_returns_default_when_key_missing(lr_model):
    assert lr_model.get("data_schema.nonexistent_field",
                        "fallback") == "fallback"


def test_get_returns_none_by_default_when_key_missing(lr_model):
    assert lr_model.get("data_schema.nonexistent_field") is None


def test_get_returns_default_when_intermediate_path_is_not_a_dict(lr_model):
    # "data_schema.target_column" resolves to a string ("target"), so trying
    # to go one level deeper should fall back to the default rather than error
    assert lr_model.get("data_schema.target_column.anything",
                        "fallback") == "fallback"


def test_get_returns_default_when_top_level_key_missing(lr_model):
    assert lr_model.get("nonexistent_top_level_key", "fallback") == "fallback"


def test_get_returns_list_value_unmodified(lr_model):
    feature_columns = lr_model.get("data_schema.feature_columns")
    assert isinstance(feature_columns, list)
    assert "age" in feature_columns


# -----------------------
# Test load_targets function
# ---------------------

@pytest.mark.parametrize("model_fixture", ["lr_model", "xgboost_model", "llm_model"])
def test_returns_correct_columns(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    id_col = model.get(config.DATA_SCHEMA_ID_COLUMN)
    target_col = model.get(config.DATA_SCHEMA_TARGET_COLUMN)
    pred_col = model.get(config.DATA_SCHEMA_PREDICTION_COLUMN)

    y_true, y_pred = model.load_targets()

    assert list(y_true.columns) == [id_col, target_col]
    assert list(y_pred.columns) == [id_col, pred_col]

@pytest.mark.parametrize("model_fixture", ["lr_model", "xgboost_model", "llm_model"])
def test_returns_matching_ids_in_same_order(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    id_col = model.get(config.DATA_SCHEMA_ID_COLUMN)

    y_true, y_pred = model.load_targets()

    assert len(y_true) == len(y_pred)
    assert list(y_true[id_col]) == list(y_pred[id_col])

def test_values_match_underlying_fake_data(lr_model):
    id_col = lr_model.get(config.DATA_SCHEMA_ID_COLUMN)
    target_col = lr_model.get(config.DATA_SCHEMA_TARGET_COLUMN)
    pred_col = lr_model.get(config.DATA_SCHEMA_PREDICTION_COLUMN)

    expected_y_true = lr_model.store.load_dataframe("y_test")[[id_col, target_col]]
    expected_y_pred = lr_model.store.load_dataframe("y_pred")[[id_col, pred_col]]

    y_true, y_pred = lr_model.load_targets()

    pd.testing.assert_frame_equal(
        y_true.reset_index(drop=True), expected_y_true.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        y_pred.reset_index(drop=True), expected_y_pred.reset_index(drop=True)
    )

def test_inner_join_drops_unmatched_ids(lr_model_factory):
    """
    y_test and y_pred only partially overlap on id -> load_targets should
    keep just the intersection (inner join), not raise or pad with NaN.
    """
    id_col, target_col, pred_col = "instance_id", "target", "prediction"

    y_test = pd.DataFrame({
        id_col: ["inst_0", "inst_1", "inst_2"],
        target_col: [0, 1, 0],
    })
    y_pred = pd.DataFrame({
        # inst_2 has no prediction, inst_3 has no ground truth
        id_col: ["inst_0", "inst_1", "inst_3"],
        pred_col: [0, 1, 1],
    })

    model = lr_model_factory(y_test=y_test, y_pred=y_pred)

    y_true_out, y_pred_out = model.load_targets()

    assert set(y_true_out[id_col]) == {"inst_0", "inst_1"}
    assert set(y_pred_out[id_col]) == {"inst_0", "inst_1"}
    assert len(y_true_out) == 2
    assert len(y_pred_out) == 2

def test_no_overlapping_ids_returns_empty(lr_model_factory):
    id_col, target_col, pred_col = "instance_id", "target", "prediction"

    y_test = pd.DataFrame({id_col: ["inst_0"], target_col: [0]})
    y_pred = pd.DataFrame({id_col: ["inst_1"], pred_col: [1]})

    model = lr_model_factory(y_test=y_test, y_pred=y_pred)

    y_true_out, y_pred_out = model.load_targets()

    assert y_true_out.empty
    assert y_pred_out.empty

def test_calls_store_with_expected_filenames(lr_model):
    lr_model.load_targets()

    lr_model.store.load_dataframe.assert_any_call("y_test")
    lr_model.store.load_dataframe.assert_any_call("y_pred")
    
    
# -----------------------
# Test get_metrics function
# ---------------------

@pytest.mark.parametrize("model_fixture", ["lr_model", "xgboost_model", "llm_model"])
def test_get_metrics_calls_store_with_metrics_key(model_fixture, request):
    model = request.getfixturevalue(model_fixture)

    model.get_metrics()

    model.store.load_series.assert_any_call("metrics")


def test_get_metrics_contains_expected_metric_names(lr_model):
    metrics = lr_model.get_metrics()

    assert "accuracy" in metrics.index
    assert config.BEST_MODEL_METRIC in metrics.index