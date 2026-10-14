"""
Tests for CoefficientsAdapter

Covers:

- explain_local_df
- explain_global_df
- Binary and multiclass types
- correct object types are saved/returned (eg dataframe)
"""



import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from concordiahub.adapters.explainers.coefficients_adapter import CoefficientsAdapter
from concordiahub.exceptions import MissingDataError
from concordiahub.entities.context import ModelData, ModelContext
from concordiahub import config
from concordiahub.entities.context import DataSchema


MODULE_UNDER_TEST = "concordiahub.adapters.explainers.coefficients_adapter"


@pytest.fixture
def adapter():
    """ create adapter object """
    return CoefficientsAdapter()

@pytest.fixture
def instance_uuids():
    """Create some instance uuids"""
    return ['uuid-001', 'uuid-002', 'uuid-003']

@pytest.fixture
def mock_model():
    """Mock the coefficients model"""
    m = MagicMock()
    m.coef_ = np.random.rand(1, 5)   # binary: one row of 5 coefficients
    return m

@pytest.fixture
def data_schema():
    """Create a data schema"""
    return DataSchema("instance_uuid", "target", "prediction")

@pytest.fixture
def x_test(instance_uuids):
    """Create a test x test"""
    df = pd.DataFrame(
        np.random.rand(3, 5),
        columns=['f1', 'f2', 'f3', 'f4', 'f5'],
    )
    df.insert(0, 'instance_uuid', instance_uuids)
    return df

@pytest.fixture
def make_ctx(data_schema):
    """Make a ModelContext"""
    def _make(x_test):
        data = ModelData(data_schema, x_test=x_test, y_test=None, raw_instances=None, y_pred=None)
        return ModelContext(model_type="logistic_regression",
                            xai_methods=["coefficients"],
                            dataset_id="dataset",
                            data=data,
                            model_display_name="model_display_name")
    return _make

def make_run(instance_id_column="id", dataframe=None):
    """Make a run (model registration"""
    run = MagicMock()
    instance_id_column = config.DATA_SCHEMA_ID_COLUMN
    run.get = lambda key, default=None: {config.DATA_SCHEMA_ID_COLUMN: instance_id_column,
        }.get(key, default)
    run.get_data_schema.return_value = DataSchema(
        instance_id_column=instance_id_column,
        target_column="target",
        prediction_column="prediction",
    )
    run.store.load_dataframe.return_value = dataframe if dataframe is not None else pd.DataFrame({
        instance_id_column: ["sample-1", "sample-2"],
        "feat1": [0.5, 0.3],
        "feat2": [0.2, 0.4],
    })
    return run


def test_returns_correct_row():
    """Test returns a correct row for a specific feature"""
    run = make_run()
    result = CoefficientsAdapter.load_local_explanation(run, "sample-1")
    row = result[result["feature_name"] == "feat1"].iloc[0]
    assert row["score"] == 0.5

def test_returns_dataframe():
    """Test a dataframe object is returned from load_local_explanation"""
    run = make_run()
    result = CoefficientsAdapter.load_local_explanation(run, "sample-1")
    assert isinstance(result, pd.DataFrame)

def test_missing_sample_raises():
    """Test an error is raised if data missing"""
    run = make_run()
    with pytest.raises(MissingDataError):
        CoefficientsAdapter.load_local_explanation(run, "nonexistent")

def test_loads_correct_dataframe():
    """Test loading a dataframe is called"""
    run = make_run()
    CoefficientsAdapter.load_local_explanation(run, "sample-1")
    run.store.load_dataframe.assert_called_once_with(f"exp_local_{CoefficientsAdapter.key}")

def test_uses_correct_id_column():
    """Test using a correct id column, and matching"""
    df = pd.DataFrame({
        config.DATA_SCHEMA_ID_COLUMN: ["sample-1"],
        "feat1": [0.5],
    })
    run = make_run(dataframe=df)
    result = CoefficientsAdapter.load_local_explanation(run, "sample-1")
    row = result[result["feature_name"] == "feat1"].iloc[0]
    assert row["score"] == 0.5


def test_global_multiclass(adapter, x_test, data_schema):
    """Test a global multiclass explanation"""
    model = MagicMock()
    model.coef_ = np.random.rand(3, 5)   # 3 classes

    data = ModelData(data_schema, x_test=x_test, y_test=None,
                     raw_instances=None, y_pred=None)
    ctx = ModelContext(model_type="logistic_regression",
                       xai_methods=["coefficients"],
                       dataset_id="dataset",
                       data=data,
                       model_display_name="model_display_name")

    result = adapter.explain_global_df(ctx, model)

    assert result.shape == (5, 2)
    assert list(result["feature"]) == ['f1', 'f2', 'f3', 'f4', 'f5']
    np.testing.assert_allclose(result["importance"], model.coef_[0])


def test_local_multiclass(adapter, x_test, data_schema):
    """Test a local multiclass explanation"""
    model = MagicMock()
    model.coef_ = np.random.rand(3, 5)   # 3 classes

    data = ModelData(data_schema, x_test=x_test, y_test=None,
                     raw_instances=None, y_pred=None)
    ctx = ModelContext(model_type="logistic_regression",
                       xai_methods=["coefficients"],
                       dataset_id="dataset",
                       data=data,
                       model_display_name="model_display_name")

    result = adapter.explain_local_df(ctx, model)

    assert result.shape == (3, 6)

def test_local_raises_when_x_test_missing(adapter, mock_model, make_ctx):
    """Test error is raised if x_test is missing"""
    ctx = make_ctx(x_test=None)

    with pytest.raises(MissingDataError) as exc:
        adapter.explain_local_df(ctx, mock_model)

    assert "x_test" in str(exc.value)

def test_local_raises_when_explain_json(adapter, mock_model):
    """Test error is raised if JSON method is used"""
    data = ModelData(data_schema, x_test=x_test, y_test=None, raw_instances=None, y_pred=None)
    ctx = ModelContext(model_type="logistic_regression",
                       xai_methods=["coefficients"],
                       dataset_id="dataset",
                       data=data,
                       model_display_name="model_display_name")

    with pytest.raises(NotImplementedError):
        adapter.explain_local_json(ctx)

