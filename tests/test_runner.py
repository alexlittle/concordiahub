import pytest
from unittest.mock import patch, MagicMock


from xailens import runner
from xailens.exceptions import InvalidConfigurationError
from xailens.entities.context import ModelContext, RunData, DataSchema

MODULE_UNDER_TEST = "xailens.runner"


# Test with no context provided
def test_runner_no_context() -> None:
    with pytest.raises(TypeError):
        runner.run()

# Test with empty context
def test_runner_empty_context() -> None:
    with pytest.raises(InvalidConfigurationError, match="Context"):
        runner.run(None)


# Test invalid model type
def test_runner_invalid_model() -> None:
    data_schema = DataSchema("a","b","c")
    run_data = RunData(data_schema)
    ctx = ModelContext("unknown", [], "test_dataset", run_data)
    with patch("xailens.runner.ArtifactStore") as mock_store:
        mock_store.return_value = MagicMock()
        with pytest.raises(InvalidConfigurationError, match="Model adapter"):
            runner.run(ctx)