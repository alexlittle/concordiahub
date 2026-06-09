import pandas as pd
import pytest

from xailens.exceptions import InvalidSelectionError
from xailens.services.comparison import build_comparison_df

MODULE_UNDER_TEST = "xailens.services.comparison"

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
        "Comparison",
        "True Value",
        "ModelA prediction",
        "ModelB prediction",
    ]

    assert result.shape == (2, 5)
    assert result["True Value"].tolist() == [10, 20]
    assert result["ModelA prediction"].tolist() == [11, 19]
    assert result["ModelB prediction"].tolist() == [9, 21]

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
        "model_type": "ModelA",
        "data_schema": {"sample_id_column": "id", "target_column": "target"}
    })

    run_b = MockRun(df_true, df_pred_b, {
        "model_type": "ModelB",
        "data_schema": {"sample_id_column": "id", "target_column": "target"}
    })

    result = build_comparison_df(run_a, run_b, mock_ensure_index).sort_values("id")

    assert result["ModelA prediction"].tolist() == [11, 19]
    assert result["ModelB prediction"].tolist() == [9, 21]

# test when missing IDs
def test_build_comparison_missing_ids():
    df_true = pd.DataFrame({
        "id": [1, 2, 3],
        "target": [10, 20, 30]
    })

    df_pred_a = pd.DataFrame({
        "id": [1, 2],
        "pred": [11, 12]
    })

    df_pred_b = pd.DataFrame({
        "id": [1, 3],
        "pred": [21, 23]
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

    # Expect no NaNs because inner join
    assert result.shape[0] == 1
    assert result["A prediction"].isna().sum() == 0
    assert result["B prediction"].isna().sum() == 0


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
    run_b = MockRun(df_true, df_pred, metadata)

    with pytest.raises(InvalidSelectionError, match="identical model names"):
        build_comparison_df(run_a, run_b, mock_ensure_index)
