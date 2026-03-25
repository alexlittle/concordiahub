
XAI_REGISTRY = {}

def register_xai(key):
    """
    Decorator to register an XAI explainer adapter by key.
    """
    def wrapper(cls):
        XAI_REGISTRY[key] = cls
        return cls
    return wrapper

def get_xai_adapter(key):
    if key not in XAI_REGISTRY:
        raise ValueError(f"XAI method '{key}' not registered.")
    return XAI_REGISTRY[key]
