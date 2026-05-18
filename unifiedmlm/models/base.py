from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

from PIL import Image


@dataclass
class VLMRequest:
    prompt: str
    images: list[Image.Image] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VLMResponse:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVLMModel(ABC):
    """Minimal interface every model must implement.

    Implementations should be batch-friendly: `generate` takes a list of
    requests and returns one response per request, in order.
    """

    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def generate(self, requests: Iterable[VLMRequest]) -> list[VLMResponse]:
        ...

    def shutdown(self) -> None:
        return None
