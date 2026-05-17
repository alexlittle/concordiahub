import pytest
from unittest.mock import patch, MagicMock

from xai_compare import runner
from xai_compare.config import RunContext

MODULE_UNDER_TEST = "xai_compare.runner"


# Test with no context provided
def test_runner_no_context() -> None:
    with pytest.raises(TypeError):
        runner.run()

# Test with empty context
def test_runner_empty_context() -> None:
    with pytest.raises(ValueError, match="Context"):
        runner.run(None)


# Test invalid model type
def test_runner_invalid_model() -> None:
    ctx = RunContext("unknown", [], "test_dataset")
    with patch("xai_compare.runner.ArtifactStore") as mock_store:
        mock_store.return_value = MagicMock()
        with pytest.raises(ValueError, match="Model adapter"):
            runner.run(ctx)