from __future__ import annotations

from typing import Any, Callable, Type

from .base import BaseVLMModel

_REGISTRY: dict[str, Type[BaseVLMModel]] = {}


def register_model(name: str) -> Callable[[Type[BaseVLMModel]], Type[BaseVLMModel]]:
    def deco(cls: Type[BaseVLMModel]) -> Type[BaseVLMModel]:
        if name in _REGISTRY:
            raise ValueError(f"Model {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def build_model(name: str, config: dict[str, Any] | None = None) -> BaseVLMModel:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](config=config)


def list_models() -> list[str]:
    return sorted(_REGISTRY)
