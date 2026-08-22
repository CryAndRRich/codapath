"""Sampler name -> function registry.

Kept in its own module so a sampler can import `register_sampler` without
importing the `sampling` package itself, which would be circular: the package
`__init__` has to import every sampler module to trigger registration.
"""

from typing import Callable, Dict, List

_SAMPLERS: Dict[str, Callable[..., List[int]]] = {}


def register_sampler(name: str):
    def wrapper(function):
        if name in _SAMPLERS:
            raise ValueError(f"Sampler '{name}' is already registered")
        _SAMPLERS[name] = function
        return function
    return wrapper


def get_sampler(name: str, **kwargs) -> List[int]:
    if name not in _SAMPLERS:
        raise ValueError(
            f"Sampler '{name}' is not registered. Available: {available_samplers()}"
        )
    return _SAMPLERS[name](**kwargs)


def available_samplers() -> List[str]:
    return sorted(_SAMPLERS)
