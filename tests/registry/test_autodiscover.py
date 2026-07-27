"""
Tests for autodiscovery of adapters

Covers:

- checking new adapters are registered correctly
- checks badly configured adapters show a warning but don't crash correct ones
- check only imports adapters once
- checks concurrency if multiple threads try to register adapters at the same time

"""
import sys
import threading
from types import ModuleType

def test_autodiscover_imports_all_submodules(monkeypatch, fresh_module):
    autodiscover = fresh_module

    # Simulate modules under each adapter type (subdirectories):
    discovered = [
        ("finder", "xailens.adapters.models.alpha", False),
        ("finder", "xailens.adapters.xai.bravo", False),
        ("finder", "xailens.adapters.viz.charlie", False),
    ]
    import_calls = []

    def fake_walk_packages(path, prefix):
        assert isinstance(path, (list, tuple))
        assert prefix == "xailens.adapters."
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
        ("finder", "xailens.adapters.good1", False),
        ("finder", "xailens.adapters.bad", False),
        ("finder", "xailens.adapters.good2", False),
    ]

    def fake_walk_packages(path, prefix):
        assert isinstance(path, (list, tuple))
        assert prefix == "xailens.adapters."
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

    # check warning printed but not a crash
    out = capsys.readouterr().out
    assert "[WARN] Failed importing xailens.adapters.bad: boom" in out

    # Good modules should still be imported
    assert "xailens.adapters.good1" in sys.modules
    assert "xailens.adapters.good2" in sys.modules


def test_autodiscover_is_idempotent(monkeypatch, fresh_module):
    autodiscover = fresh_module

    discovered_once = [
        ("finder", "xailens.adapters.once", False),
    ]

    import_calls = []

    def fake_walk_packages(path, prefix):
        assert isinstance(path, (list, tuple))
        assert prefix == "xailens.adapters."
        for item in discovered_once:
            yield item

    def fake_import_module(name):
        import_calls.append(name)
        if name not in sys.modules:
            sys.modules[name] = ModuleType(name)
        return sys.modules[name]

    monkeypatch.setattr("pkgutil.walk_packages", lambda path, prefix: fake_walk_packages(path, prefix))
    monkeypatch.setattr("importlib.import_module", fake_import_module)

    # call 1
    autodiscover.autodiscover_adapters()
    # call 2
    autodiscover.autodiscover_adapters()

    assert import_calls.count("xailens.adapters.once") == 1


def test_autodiscover_thread_safety(monkeypatch, fresh_module):
    """
    Multiple threads calling autodiscover concurrently should still only import once.
    """
    autodiscover = fresh_module

    discovered = [
        ("finder", "xailens.adapters.threaded", False),
    ]

    import_calls = []

    def fake_walk_packages(path, prefix):
        assert isinstance(path, (list, tuple))
        assert prefix == "xailens.adapters."
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

    # Even with concurrent calls, the module only imported once
    assert import_calls.count("xailens.adapters.threaded") == 1
