# tests/test_autodiscover.py
import importlib
import sys
import threading
from types import ModuleType

import pytest

# Import registry APIs from their respective modules
from xaicompare.registry.model_registry import (
    MODEL_REGISTRY,
    register_model,
    get_model_adapter,
)

from xaicompare.registry.xai_registry import (
    XAI_REGISTRY,
    register_xai,
    get_xai_adapter,
)

from xaicompare.registry.viz_registry import (
    VIZ_REGISTRY,
    register_viz,
    get_viz_adapter,
)


MODULE_UNDER_TEST = "xaicompare.registry.autodiscover"


def _ensure_clean_import():
    """
    Ensure a fresh import of the module under test by clearing it from sys.modules.
    """
    if MODULE_UNDER_TEST in sys.modules:
        del sys.modules[MODULE_UNDER_TEST]
    return importlib.import_module(MODULE_UNDER_TEST)


@pytest.fixture(autouse=True)
def fresh_module(monkeypatch):
    """
    Provides a freshly imported module for each test and ensures
    the xaicompare.adapters package exists with a __path__.
    """
    # Ensure a base 'xaicompare' package in sys.modules
    if "xaicompare" not in sys.modules:
        xaicompare = ModuleType("xaicompare")
        sys.modules["xaicompare"] = xaicompare

    # Ensure xaicompare.adapters exists and has a __path__
    adapters_pkg_name = "xaicompare.adapters"
    adapters_pkg = ModuleType(adapters_pkg_name)
    # __path__ must be a list-like to satisfy pkgutil.walk_packages
    adapters_pkg.__path__ = ["FAKE_PATH"]
    sys.modules[adapters_pkg_name] = adapters_pkg

    mod = _ensure_clean_import()
    yield mod

    # Cleanup between tests
    for key in [MODULE_UNDER_TEST, adapters_pkg_name]:
        if key in sys.modules:
            del sys.modules[key]


# ---------------------
# Model registry tests
# ---------------------
def test_register_and_get_model_adapter(fresh_module):
    @register_model("my-model")
    class MyAdapter:
        pass

    cls = get_model_adapter("my-model")
    assert cls is MyAdapter

    # Decorator returns the class unchanged
    assert MyAdapter is MODEL_REGISTRY["my-model"]


def test_get_model_adapter_missing_raises(fresh_module):
    with pytest.raises(ValueError) as exc:
        get_model_adapter("missing")
    assert "not registered" in str(exc.value)


# -------------------
# XAI registry tests
# -------------------
def test_register_and_get_xai_adapter(fresh_module):
    @register_xai("my-expl")
    class MyAdapter:
        pass

    cls = get_xai_adapter("my-expl")
    assert cls is MyAdapter

    # Decorator returns the class unchanged
    assert MyAdapter is XAI_REGISTRY["my-expl"]


def test_get_explainer_adapter_missing_raises(fresh_module):
    with pytest.raises(ValueError) as exc:
        get_xai_adapter("missing")
    assert "not registered" in str(exc.value)


# -------------------
# Viz registry tests
# -------------------
def test_register_and_get_viz_adapter(fresh_module):
    @register_viz("my-viz")
    class MyAdapter:
        pass

    cls = get_viz_adapter("my-viz")
    assert cls is MyAdapter

    # Decorator returns the class unchanged
    assert MyAdapter is VIZ_REGISTRY["my-viz"]


def test_get_viz_adapter_missing_raises(fresh_module):
    with pytest.raises(ValueError) as exc:
        get_viz_adapter("missing")
    assert "not registered" in str(exc.value)


# ---------------------------
# Autodiscovery behavior tests
# ---------------------------
def test_autodiscover_imports_all_submodules(monkeypatch, fresh_module):
    autodiscover = fresh_module

    # Simulate modules under each adapter type (subdirectories):
    discovered = [
        ("finder", "xaicompare.adapters.models.alpha", False),
        ("finder", "xaicompare.adapters.xai.bravo", False),
        ("finder", "xaicompare.adapters.viz.charlie", False),
    ]
    import_calls = []

    def fake_walk_packages(path, prefix):
        # Accept whatever real __path__ the package exposes (list/tuple of dirs)
        assert isinstance(path, (list, tuple))
        assert prefix == "xaicompare.adapters."
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
        ("finder", "xaicompare.adapters.good1", False),
        ("finder", "xaicompare.adapters.bad", False),
        ("finder", "xaicompare.adapters.good2", False),
    ]

    def fake_walk_packages(path, prefix):
        # Relaxed: do not assert path equality; only prefix
        assert isinstance(path, (list, tuple))
        assert prefix == "xaicompare.adapters."
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
    assert "[WARN] Failed importing xaicompare.adapters.bad: boom" in out

    # Good modules should still be imported (created in sys.modules)
    assert "xaicompare.adapters.good1" in sys.modules
    assert "xaicompare.adapters.good2" in sys.modules


def test_autodiscover_is_idempotent(monkeypatch, fresh_module):
    autodiscover = fresh_module

    discovered_once = [
        ("finder", "xaicompare.adapters.once", False),
    ]

    import_calls = []

    def fake_walk_packages(path, prefix):
        # Relaxed: path type + correct prefix
        assert isinstance(path, (list, tuple))
        assert prefix == "xaicompare.adapters."
        for item in discovered_once:
            yield item

    def fake_import_module(name):
        import_calls.append(name)
        if name not in sys.modules:
            sys.modules[name] = ModuleType(name)
        return sys.modules[name]

    monkeypatch.setattr("pkgutil.walk_packages", lambda path, prefix: fake_walk_packages(path, prefix))
    monkeypatch.setattr("importlib.import_module", fake_import_module)

    # First call imports
    autodiscover.autodiscover_adapters()
    # Second call should be a no-op (due to __DISCOVERED flag)
    autodiscover.autodiscover_adapters()

    assert import_calls.count("xaicompare.adapters.once") == 1


def test_autodiscover_thread_safety(monkeypatch, fresh_module):
    """
    Multiple threads calling autodiscover concurrently should still only import once.
    This exercises the lock + double-checked __DISCOVERED behavior.
    """
    autodiscover = fresh_module

    discovered = [
        ("finder", "xaicompare.adapters.threaded", False),
    ]

    import_calls = []

    def fake_walk_packages(path, prefix):
        # Relaxed: path type + correct prefix
        assert isinstance(path, (list, tuple))
        assert prefix == "xaicompare.adapters."
        # Simulate slow enumeration to increase race window
        for item in discovered:
            yield item

    def fake_import_module(name):
        import_calls.append(name)
        if name not in sys.modules:
            sys.modules[name] = ModuleType(name)
        return sys.modules[name]

    monkeypatch.setattr("pkgutil.walk_packages", lambda path, prefix: fake_walk_packages(path, prefix))
    monkeypatch.setattr("importlib.import_module", fake_import_module)

    # Call autodiscover from several threads
    threads = [threading.Thread(target=autodiscover.autodiscover_adapters) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Even with many concurrent calls, the module should be imported once
    assert import_calls.count("xaicompare.adapters.threaded") == 1