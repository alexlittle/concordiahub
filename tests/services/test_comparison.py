import pandas as pd
import pytest

from xai_compare.services.comparison import build_comparison_df

MODULE_UNDER_TEST = "xai_compare.services.comparison"

class MockRun:
    def __init__(self, y_true, y_pred, metadata):
        self._y_true = y_true
        self._y_pred = y_pred
        self.metadata = metadata

    def load_targets(self):
        return self._y_true.copy(), self._y_pred.copy()

    def get(self, key, default=None):
        parts = key.split(".")
        value = self.metadata
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

def mock_ensure_index(df, col):
    return df.set_index(col)

# test correct generation
def test_build_comparison_basic():
    df_true = pd.DataFrame({
        "id": [1, 2],
        "target": [10, 20]
    })

    df_pred_a = pd.DataFrame({
        "id": [1, 2],
        "pred": [11, 19]
    })

    df_pred_b = pd.DataFrame({
        "id": [1, 2],
        "pred": [9, 21]
    })

    run_a = MockRun(df_true, df_pred_a, {
        "model_type": "ModelA",
        "data_schema": {
            "sample_id_column": "id",
            "target_column": "target"
        }
    })

    run_b = MockRun(df_true, df_pred_b, {
        "model_type": "ModelB",
        "data_schema": {
            "sample_id_column": "id",
            "target_column": "target"
        }
    })

    result = build_comparison_df(run_a, run_b, mock_ensure_index)

    assert list(result.columns) == [
        "id",
        "True Value",
        "ModelA Prediction",
        "ModelB Prediction",
    ]

    assert result.shape == (2, 4)
    assert result["True Value"].tolist() == [10, 20]
    assert result["ModelA Prediction"].tolist() == [11, 19]
    assert result["ModelB Prediction"].tolist() == [9, 21]

# test join
def test_build_comparison_aligns_on_index():
    df_true = pd.DataFrame({
        "id": [1, 2],
        "target": [10, 20]
    })

    df_pred_a = pd.DataFrame({
        "id": [2, 1],
        "pred": [19, 11]
    })

    df_pred_b = pd.DataFrame({
        "id": [2, 1],
        "pred": [21, 9]
    })

    run_a = MockRun(df_true, df_pred_a, {
        "model_type": "A",
        "data_schema": {"sample_id_column": "id", "target_column": "target"}
    })

    run_b = MockRun(df_true, df_pred_b, {
        "model_type": "B",
        "data_schema": {"sample_id_column": "id", "target_column": "target"}
    })

    result = build_comparison_df(run_a, run_b, mock_ensure_index).sort_values("id")

    assert result["A Prediction"].tolist() == [11, 19]
    assert result["B Prediction"].tolist() == [9, 21]

# test when missing IDs
def test_build_comparison_missing_ids():
    df_true = pd.DataFrame({
        "id": [1, 2],
        "target": [10, 20]
    })

    df_pred_a = pd.DataFrame({
        "id": [1],
        "pred": [11]
    })

    df_pred_b = pd.DataFrame({
        "id": [2],
        "pred": [21]
    })

    run_a = MockRun(df_true, df_pred_a, {
        "model_type": "A",
        "data_schema": {"sample_id_column": "id", "target_column": "target"}
    })

    run_b = MockRun(df_true, df_pred_b, {
        "model_type": "B",
        "data_schema": {"sample_id_column": "id", "target_column": "target"}
    })

    result = build_comparison_df(run_a, run_b, mock_ensure_index)

    # Expect NaNs due to outer-like effect of chained joins
    assert result.shape[0] == 2
    assert result["A Prediction"].isna().sum() == 1
    assert result["B Prediction"].isna().sum() == 1


# test if same model format name
def test_build_comparison_raises_if_same_formatted_name():
    df_true = pd.DataFrame({"id": [1], "target": [5]})
    df_pred = pd.DataFrame({"id": [1], "pred": [6]})

    metadata = {
        "model_id": "123",
        "model_type": "XGBoost",
        "model_subtype": "v1",
        "data_schema": {
            "sample_id_column": "id",
            "target_column": "target"
        }
    }

    run_a = MockRun(df_true, df_pred, metadata)
    run_b = MockRun(df_true, df_pred, metadata)  # identical

    with pytest.raises(ValueError, match="identical model names"):
        build_comparison_df(run_a, run_b, mock_ensure_index)
