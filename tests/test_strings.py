"""
Tests for the helper functions in strings.py

Covers functions/class:

- recommended_title
- format_recommended_models
- DisplayNameDict

"""
import pytest

from xailens import strings

# test recommended_title
@pytest.mark.parametrize("count,expected", [
    (1, strings.RECOMMENDED_MODEL_TITLE),
    (2, strings.RECOMMENDED_MODELS_TITLE),
    (5, strings.RECOMMENDED_MODELS_TITLE),
])
def test_recommended_title(count, expected):
    assert strings.recommended_title(count) == expected


# test format_recommended_models
@pytest.mark.parametrize("models,expected", [
    (["XGBoost"], "XGBoost"),
    (["XGBoost", "Gemma 3"], "XGBoost, Gemma 3"),
    (["XGBoost", "Gemma 3", "Llama 3.1"], "XGBoost, Gemma 3, Llama 3.1"),
])
def test_format_recommended_models(models, expected):
    assert strings.format_recommended_models(models) == expected


# tests for DisplayNameDict class
@pytest.mark.parametrize("key,expected", [
    ("logistic_regression", "Logistic Regression"),
    ("xgboost", "XGBoost"),
    ("llm", "LLM"),
])
def test_model_display_name_known_keys(key, expected):
    assert strings.MODEL_DISPLAY_NAME[key] == expected


@pytest.mark.parametrize("key,expected", [
    ("random_forest", "Random Forest"),
    ("gemma_3", "Gemma 3"),
    ("single", "Single"),
])
def test_model_display_name_missing_key_fallback(key, expected):
    assert strings.MODEL_DISPLAY_NAME[key] == expected
