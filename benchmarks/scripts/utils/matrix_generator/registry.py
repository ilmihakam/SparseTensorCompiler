"""Family generator registry with @register decorator."""

from __future__ import annotations

import random
from typing import Callable, Dict, List

from .types import GeneratedMatrix, MatrixSpec

GeneratorFn = Callable[[MatrixSpec, random.Random], GeneratedMatrix]

_REGISTRY: Dict[str, GeneratorFn] = {}


def register(family_name: str):
    """Decorator to register a generator function for a family."""
    def decorator(fn: GeneratorFn) -> GeneratorFn:
        if family_name in _REGISTRY:
            raise ValueError(f"Family '{family_name}' already registered")
        _REGISTRY[family_name] = fn
        return fn
    return decorator


def get_generator(family_name: str) -> GeneratorFn:
    if family_name not in _REGISTRY:
        raise ValueError(
            f"Unknown family '{family_name}'. Available: {list_families()}"
        )
    return _REGISTRY[family_name]


def list_families() -> List[str]:
    return sorted(_REGISTRY.keys())
