from xailens.exceptions import InvalidConfigurationError

VIZ_REGISTRY = {}

def register_viz(key):
    """
    Decorator to register a viz adapter by key.
    """
    def wrapper(cls):
        cls.key = key
        VIZ_REGISTRY[key] = cls
        return cls
    return wrapper

def get_viz_adapter(key):
    if key not in VIZ_REGISTRY:
        raise InvalidConfigurationError(f"Viz adapter '{key}' not registered.")
    return VIZ_REGISTRY[key]