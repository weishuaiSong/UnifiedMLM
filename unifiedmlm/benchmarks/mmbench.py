"""MMBench loader (HuggingFace `datasets`).

Default source is `lmms-lab/MMBench` (dev split has labels). Each sample carries
image, question, options A-D (sometimes E), an `answer` letter, and category metadata.
"""
from __future__ import annotations

from typing import Any, Iterator

from PIL import Image

from .base import BaseBenchmark, BenchmarkSample
from .registry import register_benchmark


PROMPT_TEMPLATE = (
    "{question}\n"
    "{options_block}\n"
    "Answer with the letter of the correct option."
)


@register_benchmark("mmbench")
class MMBench(BaseBenchmark):
    """MMBench dev split.

    Config keys:
      hf_path:      datasets repo (default: lmms-lab/MMBench)
      hf_name:      datasets config name (default: en)
      split:        split to use (default: dev)
      limit:        optional int; if set, take only the first N samples
      shuffle_seed: optional int; shuffle before truncation
    """

    DEFAULT_HF_PATH = "lmms-lab/MMBench"
    DEFAULT_HF_NAME = "en"
    DEFAULT_SPLIT = "dev"
    OPTION_KEYS = ("A", "B", "C", "D", "E")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        from datasets import load_dataset

        ds = load_dataset(
            self.config.get("hf_path", self.DEFAULT_HF_PATH),
            self.config.get("hf_name", self.DEFAULT_HF_NAME),
            split=self.config.get("split", self.DEFAULT_SPLIT),
        )
        seed = self.config.get("shuffle_seed")
        if seed is not None:
            ds = ds.shuffle(seed=int(seed))
        limit = self.config.get("limit")
        if limit is not None:
            ds = ds.select(range(min(int(limit), len(ds))))
        self._ds = ds

    def __len__(self) -> int:
        return len(self._ds)

    def __iter__(self) -> Iterator[BenchmarkSample]:
        for row in self._ds:
            sample = self._row_to_sample(row)
            if sample is not None:
                yield sample

    def _row_to_sample(self, row: dict[str, Any]) -> BenchmarkSample | None:
        image = row.get("image")
        if image is None:
            return None
        if not isinstance(image, Image.Image):
            # Some configs return a dict {"bytes": ..., "path": ...}
            try:
                from io import BytesIO

                image = Image.open(BytesIO(image["bytes"])) if isinstance(image, dict) else image
            except Exception:
                return None
        image = image.convert("RGB")

        question = (row.get("question") or "").strip()
        hint = (row.get("hint") or "").strip() if "hint" in row else ""
        if hint:
            question = f"{hint}\n{question}"

        choices: dict[str, str] = {}
        for key in self.OPTION_KEYS:
            val = row.get(key)
            if val is None or (isinstance(val, str) and not val.strip()):
                continue
            choices[key] = str(val).strip()

        options_block = "\n".join(f"{k}. {v}" for k, v in choices.items())
        prompt = PROMPT_TEMPLATE.format(question=question, options_block=options_block)

        answer = str(row.get("answer") or "").strip().upper()
        sample_id = str(row.get("index") or row.get("id") or "")

        return BenchmarkSample(
            id=sample_id,
            prompt=prompt,
            images=[image],
            answer=answer,
            choices=choices,
            metadata={
                "category": row.get("category"),
                "l2-category": row.get("l2-category"),
                "source": row.get("source"),
            },
        )
