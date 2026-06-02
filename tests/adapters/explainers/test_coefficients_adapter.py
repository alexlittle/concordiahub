import pytest
import pandas as pd
from unittest.mock import MagicMock
from xai_compare.adapters.explainers.coefficients_adapter import CoefficientsAdapter
from xai_compare.exceptions import MissingDataError
from xai_compare import constants

def make_run(sample_id_column="id", dataframe=None):
    run = MagicMock()
    sample_id_column = constants.DATA_SCHEMA_ID_COLUMN
    run.get = lambda key, default=None: {
        constants.DATA_SCHEMA_ID_COLUMN: sample_id_column,
    }.get(key, default)

    run.store.load_dataframe.return_value = dataframe if dataframe is not None else pd.DataFrame({
        sample_id_column: ["sample-1", "sample-2"],
        "feat1": [0.5, 0.3],
        "feat2": [0.2, 0.4],
    })
    return run


def test_returns_correct_row():
    run = make_run()
    result = CoefficientsAdapter.load_explanation(run, "sample-1")
    assert result["feat1"] == 0.5

def test_returns_series():
    run = make_run()
    result = CoefficientsAdapter.load_explanation(run, "sample-1")
    assert isinstance(result, pd.Series)

def test_missing_sample_raises():
    run = make_run()
    with pytest.raises(MissingDataError):
        CoefficientsAdapter.load_explanation(run, "nonexistent")

def test_loads_correct_dataframe():
    run = make_run()
    CoefficientsAdapter.load_explanation(run, "sample-1")
    run.store.load_dataframe.assert_called_once_with(f"explanations_{CoefficientsAdapter.key}")

def test_uses_correct_id_column():
    df = pd.DataFrame({
        constants.DATA_SCHEMA_ID_COLUMN: ["sample-1"],
        "feat1": [0.5],
    })
    run = make_run(dataframe=df)
    result = CoefficientsAdapter.load_explanation(run, "sample-1")
    assert result["feat1"] == 0.5