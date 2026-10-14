"""
Fixture to create fresh module for importing adapters

"""

import importlib
import sys
from types import ModuleType

import pytest

# for autodiscover and the individual adapter registries
MODULE_UNDER_TEST = "concordiahub.registry.autodiscover"

def _ensure_clean_import():
    """
    Check a clean import is done for each test
    """
    if MODULE_UNDER_TEST in sys.modules:
        del sys.modules[MODULE_UNDER_TEST]
    return importlib.import_module(MODULE_UNDER_TEST)

@pytest.fixture(autouse=True)
def fresh_module(monkeypatch):
    """
    Provides a new imported module for each test
    """
    #  'concordiahub' package in sys.modules
    if "concordiahub" not in sys.modules:
        concordiahub = ModuleType("concordiahub")
        sys.modules["concordiahub"] = concordiahub

    # concordiahub.adapters exists and has a __path__
    adapters_pkg_name = "concordiahub.adapters"
    adapters_pkg = ModuleType(adapters_pkg_name)
    adapters_pkg.__path__ = ["FAKE_PATH"]
    sys.modules[adapters_pkg_name] = adapters_pkg

    mod = _ensure_clean_import()
    yield mod

    # Cleanup between tests
    for key in [MODULE_UNDER_TEST, adapters_pkg_name]:
        if key in sys.modules:
            del sys.modules[key]