from __future__ import annotations

from typing import Any, Callable, Type

from .base import BaseBenchmark

_REGISTRY: dict[str, Type[BaseBenchmark]] = {}


def register_benchmark(name: str) -> Callable[[Type[BaseBenchmark]], Type[BaseBenchmark]]:
    def deco(cls: Type[BaseBenchmark]) -> Type[BaseBenchmark]:
        if name in _REGISTRY:
            raise ValueError(f"Benchmark {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def build_benchmark(name: str, config: dict[str, Any] | None = None) -> BaseBenchmark:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown benchmark {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](config=config)


def list_benchmarks() -> list[str]:
    return sorted(_REGISTRY)
