"""
Tests for adapters.models.model_base

Covers:
- init
- get model
"""

from unittest.mock import MagicMock

from xailens.adapters.models.model_base import ModelAdapter

def test_init_stores_model():
    """Tests the __init__ method of the ModelAdapter"""
    model = MagicMock(name="model")
    adapter = ModelAdapter(model)
    assert adapter.model is model


def test_get_model_returns_the_model():
    """Tests the get_model method of the ModelAdapter"""
    model = MagicMock(name="model")
    adapter = ModelAdapter(model)
    assert adapter.get_model() is model


def test_model_defaults_to_none():
    """Tests the get_model method of the ModelAdapter"""
    assert ModelAdapter().get_model() is None