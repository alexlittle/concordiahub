
from xai_compare.exceptions import InvalidConfigurationError

MODEL_REGISTRY = {}

def register_model(key):
    """
    Decorator to register a model adapter by key.
    """
    def wrapper(cls):
        cls.key = key
        MODEL_REGISTRY[key] = cls
        return cls
    return wrapper

def get_model_adapter(key, model=None):
    if key not in MODEL_REGISTRY:
        raise InvalidConfigurationError(f"Model adapter '{key}' not registered.")
    return MODEL_REGISTRY[key](model)
