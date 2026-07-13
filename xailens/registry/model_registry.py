"""
ModelRegistry
"""


from typing import Callable, Type, TypeVar

from xailens.exceptions import InvalidConfigurationError
from xailens.adapters.models.model_base import ModelAdapter

T = TypeVar("T", bound=ModelAdapter)


MODEL_REGISTRY: dict[str, Type[ModelAdapter]] = {}

def register_model(key: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a model adapter by key.

    Args:
        key (str): model adapter key

    Returns:
        Callable[[Type[T]], Type[T]]: decorator that registers and returns
            the decorated class unchanged
    """
    def wrapper(cls: Type[T]) -> Type[T]:
        """
        Registers the given class in the model registry.

        Args:
            cls (Type[T]): class being registered

        Returns:
            Type[T]: the same class, unmodified other than the added key attribute
        """
        cls.key = key
        MODEL_REGISTRY[key] = cls
        return cls
    return wrapper

def get_model_adapter(key: str)-> Type[ModelAdapter]:
    """
    Retrieves the registered model adapter class for the given key.

    Args:
       key (str): model adapter key

    Returns:
       Type[ModelAdapter]: the registered adapter class

    Raises:
       InvalidConfigurationError: if no adapter is registered under key
    """
    if key not in MODEL_REGISTRY:
        raise InvalidConfigurationError(f"Model adapter '{key}' not registered.")
    return MODEL_REGISTRY[key]
