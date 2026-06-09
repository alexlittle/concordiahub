from xailens.exceptions import InvalidConfigurationError

XAI_REGISTRY = {}

def register_xai(key):
    """
    Decorator to register an XAI explainer adapter by key.
    """
    def wrapper(cls):
        cls.key = key
        XAI_REGISTRY[key] = cls
        return cls
    return wrapper

def get_xai_adapter(key):
    if key not in XAI_REGISTRY:
        raise InvalidConfigurationError(f"XAI method '{key}' not registered.")
    return XAI_REGISTRY[key]
