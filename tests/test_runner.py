"""
Tests for runner.py, specifically a call to the run function.

Covers:
- no context provided
- empty context provided
- completes successfully
"""


import pytest
import dataclasses

from unittest.mock import patch

from xailens import runner
from xailens.exceptions import InvalidConfigurationError, MissingDataError

MODULE_UNDER_TEST = "xailens.runner"


# Test with no context provided
def test_runner_no_context() -> None:
    with pytest.raises(TypeError):
        runner.run()

# Test with empty context
def test_runner_empty_context() -> None:
    with pytest.raises(InvalidConfigurationError, match="Context"):
        runner.run(None)

# test runs correctly with dataframe format
def test_runner_correct_df(model_context_factory):
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


# test runs correctly with JSON explanation format
def test_runner_correct_json(model_context_factory):
    ctx = model_context_factory(xai_methods=["llm_reasoning"])

    with patch(f"{MODULE_UNDER_TEST}.ArtifactStore") as MockArtifactStore, \
         patch(f"{MODULE_UNDER_TEST}.get_model_adapter") as mock_get_model_adapter, \
         patch(f"{MODULE_UNDER_TEST}.get_xai_adapter") as mock_get_xai_adapter:

        mock_store = MockArtifactStore.return_value

        mock_xai_adapter_class = mock_get_xai_adapter.return_value
        mock_xai_adapter = mock_xai_adapter_class.return_value
        mock_xai_adapter.explanation_file_type = "json"
        mock_xai_adapter.explain_global_json.return_value = "fake_global_json"
        mock_xai_adapter.explain_local_json.return_value = "fake_local_json"

        runner.run(ctx)

        mock_store.write_metrics.assert_called_once()
        mock_store.write_metadata.assert_called_once()
        mock_store.write_model.assert_called_once()
        mock_store.write_explanation.assert_any_call(
            "exp_global_llm_reasoning", "fake_global_json", data_format="df"
        )
        mock_store.write_explanation.assert_any_call(
            "exp_local_llm_reasoning", "fake_local_json", data_format="json"
        )


# test runner with missing y_test
def test_runner_missing_y_test(model_context_factory, model_data_factory):
    data = dataclasses.replace(model_data_factory(), y_test=None)
    ctx = model_context_factory(data=data)
    with pytest.raises(MissingDataError):
        runner.compute_metrics(ctx.data)

# test runner with missing y_pred
def test_runner_missing_y_pred(model_context_factory, model_data_factory):
    data = dataclasses.replace(model_data_factory(), y_pred=None)
    ctx = model_context_factory(data=data)
    with pytest.raises(MissingDataError):
        runner.compute_metrics(ctx.data)
