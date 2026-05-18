from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..benchmarks import build_benchmark
from ..models import build_model, VLMRequest
from .extractor import TwoStepExtractor


@dataclass
class EvalResult:
    model: str
    benchmark: str
    accuracy: float
    n_total: int
    n_correct: int
    n_extracted: int
    per_sample: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class EvalRunner:
    def __init__(
        self,
        model_name: str,
        benchmark_name: str,
        model_cfg: dict[str, Any] | None = None,
        benchmark_cfg: dict[str, Any] | None = None,
        extractor_cfg: dict[str, Any] | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.benchmark_name = benchmark_name
        self.model_cfg = model_cfg or {}
        self.benchmark_cfg = benchmark_cfg or {}
        self.extractor_cfg = extractor_cfg or {}
        self.batch_size = batch_size

    def run(self, output_dir: str | Path | None = None) -> EvalResult:
        benchmark = build_benchmark(self.benchmark_name, self.benchmark_cfg)
        model = build_model(self.model_name, self.model_cfg)
        extractor = TwoStepExtractor(**self.extractor_cfg)

        samples = list(benchmark)
        total = len(samples)
        per_sample: list[dict[str, Any]] = []
        n_correct = 0
        n_extracted = 0
        t0 = time.time()

        for start in tqdm(range(0, total, self.batch_size), desc=f"{self.model_name}/{self.benchmark_name}"):
            chunk = samples[start : start + self.batch_size]
            requests = [VLMRequest(prompt=s.prompt, images=s.images, metadata={"id": s.id}) for s in chunk]
            responses = model.generate(requests)
            for sample, resp in zip(chunk, responses):
                if benchmark.task_type == "mcq":
                    ext = extractor.extract_mcq(resp.text, sample.choices or {})
                else:
                    ext = extractor.extract_open(resp.text, sample.answer)
                correct = bool(ext.predicted) and ext.predicted.strip().upper() == sample.answer.strip().upper()
                if correct:
                    n_correct += 1
                if ext.predicted:
                    n_extracted += 1
                per_sample.append(
                    {
                        "id": sample.id,
                        "prompt": sample.prompt,
                        "gold": sample.answer,
                        "raw": resp.text,
                        "predicted": ext.predicted,
                        "extraction_method": ext.method,
                        "extraction_score": ext.score,
                        "correct": correct,
                        "category": sample.metadata.get("category"),
                    }
                )

        model.shutdown()
        elapsed = time.time() - t0
        accuracy = n_correct / total if total else 0.0
        result = EvalResult(
            model=self.model_name,
            benchmark=self.benchmark_name,
            accuracy=accuracy,
            n_total=total,
            n_correct=n_correct,
            n_extracted=n_extracted,
            per_sample=per_sample,
            meta={
                "elapsed_sec": elapsed,
                "batch_size": self.batch_size,
                "model_cfg": self.model_cfg,
                "benchmark_cfg": self.benchmark_cfg,
                "extractor_cfg": self.extractor_cfg,
            },
        )

        if output_dir is not None:
            self._dump(result, Path(output_dir))
        return result

    @staticmethod
    def _dump(result: EvalResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": result.model,
            "benchmark": result.benchmark,
            "accuracy": result.accuracy,
            "n_total": result.n_total,
            "n_correct": result.n_correct,
            "n_extracted": result.n_extracted,
            "meta": result.meta,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        with (output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as f:
            for row in result.per_sample:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
