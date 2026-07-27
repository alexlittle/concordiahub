"""
Tests for xai_registry:

covers:

- correctly registering a new xai adapter
- InvalidConfigurationError is raised if invalid adapter requested
"""

import pytest

from xailens.exceptions import InvalidConfigurationError

from xailens.registry.xai_registry import (
    XAI_REGISTRY,
    register_xai,
    get_xai_adapter,
)

def test_register_and_get_xai_adapter(fresh_module):
    @register_xai("my-expl")
    class MyAdapter:
        def __init__(self):
            pass

    xai_class = get_xai_adapter("my-expl")
    xai_adapter = xai_class()
    assert isinstance(xai_adapter, MyAdapter)
    assert MyAdapter is XAI_REGISTRY["my-expl"]


def test_get_explainer_adapter_missing_raises(fresh_module):
    with pytest.raises(InvalidConfigurationError) as exc:
        get_xai_adapter("missing")
    assert "not registered" in str(exc.value)