
import importlib
import sys
import threading
from types import ModuleType

import pytest

from xai_compare.exceptions import InvalidConfigurationError

# Import registry APIs from their respective modules
from xai_compare.registry.model_registry import (
    MODEL_REGISTRY,
    register_model,
    get_model_adapter,
)

from xai_compare.registry.xai_registry import (
    XAI_REGISTRY,
    register_xai,
    get_xai_adapter,
)

from xai_compare.registry.viz_registry import (
    VIZ_REGISTRY,
    register_viz,
    get_viz_adapter,
)

from xai_compare.registry.comparison_registry import (
    COMPARISON_REGISTRY,
    register_comparison,
    get_comparison_adapter,
)


MODULE_UNDER_TEST = "xai_compare.registry.autodiscover"


def _ensure_clean_import():
    """
    Ensure a fresh import by clearing it from sys.modules.
    """
    if MODULE_UNDER_TEST in sys.modules:
        del sys.modules[MODULE_UNDER_TEST]
    return importlib.import_module(MODULE_UNDER_TEST)


@pytest.fixture(autouse=True)
def fresh_module(monkeypatch):
    """
    Provides a freshly imported module for each test and ensures
    the xai_compare.adapters package exists with a __path__.
    """
    #  'xai_compare' package in sys.modules
    if "xai_compare" not in sys.modules:
        xai_compare = ModuleType("xai_compare")
        sys.modules["xai_compare"] = xai_compare

    # xai_compare.adapters exists and has a __path__
    adapters_pkg_name = "xai_compare.adapters"
    adapters_pkg = ModuleType(adapters_pkg_name)
    adapters_pkg.__path__ = ["FAKE_PATH"]
    sys.modules[adapters_pkg_name] = adapters_pkg

    mod = _ensure_clean_import()
    yield mod

    # Cleanup between tests
    for key in [MODULE_UNDER_TEST, adapters_pkg_name]:
        if key in sys.modules:
            del sys.modules[key]

# ------------------------
# Model registry tests
# ------------------------
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


# ------------------------
# XAI registry tests
# ------------------------
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


# ------------------------
# Viz registry tests
# ------------------------
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

# ------------------------
# Comparison registry tests
# ------------------------
def test_register_and_get_comparison_adapter(fresh_module):
    @register_comparison("method-a", "method-b")
    class MyAdapter:
        pass

    comp_class = get_comparison_adapter("method-a", "method-b")
    comp_adapter = comp_class()
    assert isinstance(comp_adapter, MyAdapter)

    # Check that the symmetry works
    comp_class_reversed = get_comparison_adapter("method-b", "method-a")
    comp_adapter_reversed = comp_class_reversed()
    assert isinstance(comp_adapter_reversed, MyAdapter)

    assert MyAdapter is COMPARISON_REGISTRY[frozenset({"method-a", "method-b"})]


def test_get_comparison_missing_raises(fresh_module):
    with pytest.raises(InvalidConfigurationError) as exc:
        get_comparison_adapter("missing_a", "missing_b")
    assert "No comparison adapter registered" in str(exc.value)

# ------------------------
# Autodiscovery tests
# ------------------------
def test_autodiscover_imports_all_submodules(monkeypatch, fresh_module):
    autodiscover = fresh_module

    # Simulate modules under each adapter type (subdirectories):
    discovered = [
        ("finder", "xai_compare.adapters.models.alpha", False),
        ("finder", "xai_compare.adapters.xai.bravo", False),
        ("finder", "xai_compare.adapters.viz.charlie", False),
    ]
    import_calls = []

    def fake_walk_packages(path, prefix):
        assert isinstance(path, (list, tuple))
        assert prefix == "xai_compare.adapters."
        yield from discovered

    def fake_import_module(name):
        import_calls.append(name)
        if name not in sys.modules:
            sys.modules[name] = ModuleType(name)
        return sys.modules[name]

    monkeypatch.setattr("pkgutil.walk_packages", lambda path, prefix: fake_walk_packages(path, prefix))
    monkeypatch.setattr("importlib.import_module", fake_import_module)

    autodiscover.autodiscover_adapters()
    assert set(import_calls) == {m[1] for m in discovered}


def test_autodiscover_continues_on_import_error(monkeypatch, fresh_module, capsys):
    autodiscover = fresh_module

    discovered = [
        ("finder", "xai_compare.adapters.good1", False),
        ("finder", "xai_compare.adapters.bad", False),
        ("finder", "xai_compare.adapters.good2", False),
    ]

    def fake_walk_packages(path, prefix):
        assert isinstance(path, (list, tuple))
        assert prefix == "xai_compare.adapters."
        for item in discovered:
            yield item

    def fake_import_module(name):
        if name.endswith(".bad"):
            raise RuntimeError("boom")
        if name not in sys.modules:
            sys.modules[name] = ModuleType(name)
        return sys.modules[name]

    monkeypatch.setattr("pkgutil.walk_packages", lambda path, prefix: fake_walk_packages(path, prefix))
    monkeypatch.setattr("importlib.import_module", fake_import_module)

    autodiscover.autodiscover_adapters()

    # Verify warning printed but function did not raise
    out = capsys.readouterr().out
    assert "[WARN] Failed importing xai_compare.adapters.bad: boom" in out

    # Good modules should still be imported
    assert "xai_compare.adapters.good1" in sys.modules
    assert "xai_compare.adapters.good2" in sys.modules


def test_autodiscover_is_idempotent(monkeypatch, fresh_module):
    autodiscover = fresh_module

    discovered_once = [
        ("finder", "xai_compare.adapters.once", False),
    ]

    import_calls = []

    def fake_walk_packages(path, prefix):
        # Relaxed: path type + correct prefix
        assert isinstance(path, (list, tuple))
        assert prefix == "xai_compare.adapters."
        for item in discovered_once:
            yield item

    def fake_import_module(name):
        import_calls.append(name)
        if name not in sys.modules:
            sys.modules[name] = ModuleType(name)
        return sys.modules[name]

    monkeypatch.setattr("pkgutil.walk_packages", lambda path, prefix: fake_walk_packages(path, prefix))
    monkeypatch.setattr("importlib.import_module", fake_import_module)

    # First call
    autodiscover.autodiscover_adapters()
    # Second call
    autodiscover.autodiscover_adapters()

    assert import_calls.count("xai_compare.adapters.once") == 1


def test_autodiscover_thread_safety(monkeypatch, fresh_module):
    """
    Multiple threads calling autodiscover concurrently should still only import once.
    This exercises the lock + double-checked __DISCOVERED behavior.
    """
    autodiscover = fresh_module

    discovered = [
        ("finder", "xai_compare.adapters.threaded", False),
    ]

    import_calls = []

    def fake_walk_packages(path, prefix):
        assert isinstance(path, (list, tuple))
        assert prefix == "xai_compare.adapters."
        for item in discovered:
            yield item

    def fake_import_module(name):
        import_calls.append(name)
        if name not in sys.modules:
            sys.modules[name] = ModuleType(name)
        return sys.modules[name]

    monkeypatch.setattr("pkgutil.walk_packages", lambda path, prefix: fake_walk_packages(path, prefix))
    monkeypatch.setattr("importlib.import_module", fake_import_module)

    threads = [threading.Thread(target=autodiscover.autodiscover_adapters) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Even with many concurrent calls, the module should be imported once
    assert import_calls.count("xai_compare.adapters.threaded") == 1