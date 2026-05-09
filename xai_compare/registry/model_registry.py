
MODEL_REGISTRY = {}

def register_model(key):
    """
    Decorator to register a model adapter by key.
    """
    def wrapper(cls):
        MODEL_REGISTRY[key] = cls
        return cls
    return wrapper

def get_model_adapter(key):
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Model adapter '{key}' not registered.")
    return MODEL_REGISTRY[key]
