# ./registry/autodiscover.py
import importlib
import pkgutil
import threading
import xai_compare.adapters as adapters_pkg

__DISCOVERED = False
__LOCK = threading.Lock()

def autodiscover_adapters():
    """
    for @register_model / @register_xai / @register_viz
    """
    global __DISCOVERED
    if __DISCOVERED:
        return
    with __LOCK:
        if __DISCOVERED:
            return

        for finder, modname, ispkg in pkgutil.walk_packages(
            adapters_pkg.__path__, adapters_pkg.__name__ + "."
        ):
            try:
                importlib.import_module(modname)
            except Exception as e:
                # Don't fail if one module fails
                print(f"[WARN] Failed importing {modname}: {e}")

        __DISCOVERED = True
