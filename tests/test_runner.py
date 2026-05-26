import pytest
from unittest.mock import patch, MagicMock


from xai_compare import runner
from xai_compare.exceptions import InvalidConfigurationError
from xai_compare.config import RunContext, RunData, DataSchema

MODULE_UNDER_TEST = "xai_compare.runner"


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
    data_schema = DataSchema("a","b")
    run_data = RunData(data_schema)
    ctx = RunContext("unknown", [], "test_dataset", run_data)
    with patch("xai_compare.runner.ArtifactStore") as mock_store:
        mock_store.return_value = MagicMock()
        with pytest.raises(InvalidConfigurationError, match="Model adapter"):
            runner.run(ctx)