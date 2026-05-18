from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

from PIL import Image


@dataclass
class BenchmarkSample:
    """One evaluation example.

    Attributes:
        id:        Stable per-benchmark identifier.
        prompt:    Text prompt fed to the model (already includes choices if MCQ).
        images:    List of PIL images. Most benchmarks have exactly one.
        answer:    Ground-truth answer string. For MCQ this is the letter ("A").
        choices:   Optional mapping of letter -> option text, for MCQ.
        metadata:  Free-form (category, source split, etc.).
    """

    id: str
    prompt: str
    images: list[Image.Image]
    answer: str
    choices: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseBenchmark(ABC):
    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    @abstractmethod
    def __iter__(self) -> Iterator[BenchmarkSample]:
        ...

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def task_type(self) -> str:
        """One of {'mcq', 'open'}. Drives extractor selection."""
        return self.config.get("task_type", "mcq")
