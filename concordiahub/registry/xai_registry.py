"""
XAI Method Registry
"""

from typing import Callable, Type, TypeVar

from concordiahub.adapters.explainers.xaimethod_base import XAIMethodAdapter
from concordiahub.exceptions import InvalidConfigurationError

T = TypeVar("T", bound=XAIMethodAdapter)

XAI_REGISTRY: dict[str, Type[XAIMethodAdapter]] = {}

def register_xai(key: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register an XAI method adapter by key.

    Args:
        key (str): XAI method adapter key

    Returns:
        Callable[[Type[T]], Type[T]]: decorator that registers and returns
            the decorated class unchanged
    """

    def wrapper(cls: Type[T]) -> Type[T]:
        """
        Registers the given class in the viz registry.

        Args:
            cls (Type[T]): class being registered

        Returns:
            Type[T]: the same class, unmodified other than the added key attribute
        """
        cls.key = key
        XAI_REGISTRY[key] = cls
        return cls
    return wrapper

def get_xai_adapter(key: str)-> Type[XAIMethodAdapter]:
    """
    Retrieves the registered XAI method adapter class for the given key.

    Args:
       key (str): XAI method adapter key

    Returns:
       Type[XAIMethodAdapter]: the registered adapter class

    Raises:
       InvalidConfigurationError: if no adapter is registered under key
    """

    if key not in XAI_REGISTRY:
        raise InvalidConfigurationError(f"XAI method '{key}' not registered.")
    return XAI_REGISTRY[key]
