"""
VizRegistry
"""

from typing import Callable, Type, TypeVar

from xailens.adapters.viz.viz_base import VizAdapter
from xailens.exceptions import InvalidConfigurationError

VIZ_REGISTRY = {}

T = TypeVar("T", bound=VizAdapter)

VIZ_REGISTRY: dict[str, Type[VizAdapter]] = {}

def register_viz(key: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a viz adapter by key.

    Args:
        key (str): viz adapter key

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
        VIZ_REGISTRY[key] = cls
        return cls
    return wrapper

def get_viz_adapter(key: str)-> Type[VizAdapter]:
    """
    Retrieves the registered viz adapter class for the given key.

    Args:
       key (str): viz adapter key

    Returns:
       Type[VizAdapter]: the registered adapter class

    Raises:
       InvalidConfigurationError: if no adapter is registered under key
    """
    if key not in VIZ_REGISTRY:
        raise InvalidConfigurationError(f"Viz adapter '{key}' not registered.")
    return VIZ_REGISTRY[key]