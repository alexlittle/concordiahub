"""
Tests for model_registry:

covers:

- correctly registering a new model adapter
- InvalidConfigurationError is raised if invalid adapter requested
"""

import pytest

from concordiahub.exceptions import InvalidConfigurationError

from concordiahub.registry.model_registry import (
    MODEL_REGISTRY,
    register_model,
    get_model_adapter,
)



def test_register_and_get_model_adapter(fresh_module):
    @register_model("my-model")
    class MyAdapter:
        def __init__(self, model=None):
            self.model = model

    model_class = get_model_adapter("my-model")
    model_adapter = model_class(None)
    assert isinstance(model_adapter, MyAdapter)
    assert MyAdapter is MODEL_REGISTRY["my-model"]


def test_get_model_adapter_missing_raises(fresh_module):
    with pytest.raises(InvalidConfigurationError) as exc:
        get_model_adapter("missing")
    assert "not registered" in str(exc.value)