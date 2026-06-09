from xailens.exceptions import InvalidConfigurationError

COMPARISON_REGISTRY = {}

def register_comparison(*method_keys: str):
    """
    Note that order doesn't matter — frozenset makes symmetric
    """
    def wrapper(cls):
        key = frozenset(method_keys)
        cls.key = key
        COMPARISON_REGISTRY[key] = cls
        return cls
    return wrapper

def get_comparison_adapter(method_a: str, method_b: str):
    key = frozenset({method_a, method_b})
    if key not in COMPARISON_REGISTRY:
        raise InvalidConfigurationError(
            f"No comparison adapter registered for '{method_a}' vs '{method_b}'."
        )
    return COMPARISON_REGISTRY[key]