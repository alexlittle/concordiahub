import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from xailens import runner
from xailens.exceptions import InvalidConfigurationError, MissingDataError
from xailens.entities.context import ModelContext, ModelData, DataSchema

MODULE_UNDER_TEST = "xailens.runner"


# Test with no context provided
def test_runner_no_context() -> None:
    with pytest.raises(TypeError):
        runner.run()

# Test with empty context
def test_runner_empty_context() -> None:
    with pytest.raises(InvalidConfigurationError, match="Context"):
        runner.run(None)

# test runs correctly
def test_runner_correct(model_context_factory):
    ctx = model_context_factory(xai_methods=["coefficients"])

    with patch(f"{MODULE_UNDER_TEST}.ArtifactStore") as MockArtifactStore, \
         patch(f"{MODULE_UNDER_TEST}.get_model_adapter") as mock_get_model_adapter, \
         patch(f"{MODULE_UNDER_TEST}.get_xai_adapter") as mock_get_xai_adapter:

        mock_store = MockArtifactStore.return_value
        mock_model_adapter_class = mock_get_model_adapter.return_value
        mock_xai_adapter_class = mock_get_xai_adapter.return_value
        mock_xai_adapter = mock_xai_adapter_class.return_value
        mock_xai_adapter.explanation_file_type = "df"
        mock_xai_adapter.explain_global_df.return_value = "fake_global_df"
        mock_xai_adapter.explain_local_df.return_value = "fake_local_df"

        runner.run(ctx)

        mock_store.write_metrics.assert_called_once()
        mock_store.write_metadata.assert_called_once()
        mock_store.write_model.assert_called_once()
        mock_store.write_explanation.assert_any_call(
            "exp_global_coefficients", "fake_global_df", data_format="df"
        )
        mock_store.write_explanation.assert_any_call(
            "exp_local_coefficients", "fake_local_df", data_format="df"
        )
