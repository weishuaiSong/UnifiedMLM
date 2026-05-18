"""Command-line entry point.

Usage:
    unifiedmlm-eval --config configs/eval/llava15_mmbench.yaml
    unifiedmlm-eval --model llava-1.5-7b --benchmark mmbench --limit 50
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .eval import EvalRunner
from .models import list_models
from .benchmarks import list_benchmarks


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="unifiedmlm-eval")
    parser.add_argument("--config", type=str, default=None, help="YAML eval config (overrides other flags).")
    parser.add_argument("--model", type=str, default=None, help=f"Registered model. One of: {list_models()}")
    parser.add_argument("--benchmark", type=str, default=None, help=f"Registered benchmark. One of: {list_benchmarks()}")
    parser.add_argument("--limit", type=int, default=None, help="Truncate benchmark to first N samples.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default=None, help="Where to dump summary.json + per_sample.jsonl.")
    parser.add_argument("--list", action="store_true", help="List registered models & benchmarks and exit.")
    args = parser.parse_args(argv)

    if args.list:
        print("models:    ", ", ".join(list_models()))
        print("benchmarks:", ", ".join(list_benchmarks()))
        return 0

    cfg: dict[str, Any] = {}
    if args.config:
        cfg = _load_yaml(args.config)

    model_name = cfg.get("model", {}).get("name") or args.model
    benchmark_name = cfg.get("benchmark", {}).get("name") or args.benchmark
    if not model_name or not benchmark_name:
        parser.error("Need --model and --benchmark (or --config).")

    model_cfg = cfg.get("model", {}).get("config", {}) or {}
    benchmark_cfg = dict(cfg.get("benchmark", {}).get("config", {}) or {})
    if args.limit is not None:
        benchmark_cfg["limit"] = args.limit
    extractor_cfg = cfg.get("extractor", {}) or {}
    batch_size = cfg.get("batch_size", args.batch_size)
    output_dir = args.output_dir or cfg.get("output_dir")

    runner = EvalRunner(
        model_name=model_name,
        benchmark_name=benchmark_name,
        model_cfg=model_cfg,
        benchmark_cfg=benchmark_cfg,
        extractor_cfg=extractor_cfg,
        batch_size=batch_size,
    )
    result = runner.run(output_dir=output_dir)

    print(
        f"\n[{result.model} @ {result.benchmark}] "
        f"accuracy = {result.accuracy:.4f}  "
        f"({result.n_correct}/{result.n_total}, extracted={result.n_extracted})  "
        f"elapsed={result.meta.get('elapsed_sec', 0):.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
