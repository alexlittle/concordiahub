VIZ_REGISTRY = {}

def register_viz(key):
    """
    Decorator to register a viz adapter by key.
    """
    def wrapper(cls):
        VIZ_REGISTRY[key] = cls
        return cls
    return wrapper

def get_viz_adapter(key):
    if key not in VIZ_REGISTRY:
        raise ValueError(f"Viz adapter '{key}' not registered.")
    return VIZ_REGISTRY[key]