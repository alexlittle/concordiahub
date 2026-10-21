"""
Tests for viz_registry:

covers:

- correctly registering a new viz adapter
- InvalidConfigurationError is raised if invalid adapter requested
"""

import pytest

from concordiahub.exceptions import InvalidConfigurationError


from concordiahub.registry.viz_registry import (
    VIZ_REGISTRY,
    register_viz,
    get_viz_adapter,
)


def test_register_and_get_viz_adapter(fresh_module):
    @register_viz("my-viz")
    class MyAdapter:
        pass

    cls = get_viz_adapter("my-viz")
    assert cls is MyAdapter

    # Decorator returns the class unchanged
    assert MyAdapter is VIZ_REGISTRY["my-viz"]


def test_get_viz_adapter_missing_raises(fresh_module):
    with pytest.raises(InvalidConfigurationError) as exc:
        get_viz_adapter("missing")
    assert "not registered" in str(exc.value)