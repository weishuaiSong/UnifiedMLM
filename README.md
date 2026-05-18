# UnifiedMLM

Unified evaluation (and, later, training) pipeline for multimodal LLMs.
The evaluation side is vLLM-accelerated and registry-based so new models and
benchmarks plug in without touching the runner.

This repo is the engineering backbone for the template-scaling work in
`2412.08307` and its extensions — it standardizes the eval口径 so every
experiment (Scaling-Law, Test-Train Distance, Local Competition, …) reports
numbers that are directly comparable.

> Current status: **v0.0.1** — vLLM eval line only, one model wired up
> (LLaVA-1.5-7B), one benchmark wired up (MMBench-en/dev). Training (LLaMA-Factory)
> will land in a sibling subtree later.

---

## 1. Layout

```
UnifiedMLM/
├── unifiedmlm/
│   ├── models/             # model wrappers (BaseVLMModel + registry)
│   │   ├── base.py
│   │   ├── registry.py
│   │   └── llava.py        # LLaVA-1.5-7B via vLLM
│   ├── benchmarks/         # benchmark loaders (BaseBenchmark + registry)
│   │   ├── base.py
│   │   ├── registry.py
│   │   └── mmbench.py      # MMBench (HuggingFace `lmms-lab/MMBench`)
│   ├── eval/
│   │   ├── extractor.py    # two-step answer extraction (rule → sbert)
│   │   └── runner.py       # orchestrator + result dump
│   └── cli.py              # `unifiedmlm-eval` entry point
├── configs/
│   └── eval/
│       └── llava15_mmbench.yaml
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 2. Install

```bash
# inside a fresh CUDA-enabled env (Python ≥ 3.10)
pip install -r requirements.txt
pip install -e .
```

vLLM brings its own torch wheel; if you have a pinned CUDA, follow the
[vLLM install guide](https://docs.vllm.ai/en/latest/getting_started/installation.html).

## 3. Smoke test (100 samples)

```bash
unifiedmlm-eval \
  --model llava-1.5-7b \
  --benchmark mmbench \
  --limit 100 \
  --batch-size 16 \
  --output-dir outputs/smoke_llava15_mmbench
```

Output:
- `outputs/.../summary.json` — accuracy, counts, elapsed, configs
- `outputs/.../per_sample.jsonl` — one row per example with `gold`, `raw`,
  `predicted`, `extraction_method`, `extraction_score`, `correct`

## 4. Full run via YAML

```bash
unifiedmlm-eval --config configs/eval/llava15_mmbench.yaml
```

Edit `configs/eval/llava15_mmbench.yaml` to switch `tensor_parallel`,
sampling, MMBench split, sentence-transformer model, etc.

## 5. Extending

### Add a new model

1. Create `unifiedmlm/models/<name>.py`.
2. Subclass `BaseVLMModel` and decorate with `@register_model("<name>")`.
3. Implement `generate(requests) -> list[VLMResponse]`. Use the model's own
   prompt format (LLaVA-1.5 uses `USER: <image>\n{q}\nASSISTANT:`; others differ).
4. Import the new module in `unifiedmlm/models/__init__.py` so the decorator runs.

### Add a new benchmark

1. Create `unifiedmlm/benchmarks/<name>.py`.
2. Subclass `BaseBenchmark`, decorate with `@register_benchmark("<name>")`.
3. Implement `__iter__` yielding `BenchmarkSample(id, prompt, images, answer, choices, metadata)`.
4. Set `task_type` to `"mcq"` or `"open"` (drives extractor selection).
5. Import it in `unifiedmlm/benchmarks/__init__.py`.

### Custom extractor

`TwoStepExtractor` is the default. If a benchmark needs domain-specific
parsing (e.g. numeric tolerance for MathVista), subclass it and inject via
the runner's `extractor_cfg` — or add a separate extractor and switch in
the runner based on `benchmark.task_type`.

## 6. Evaluation口径 alignment

The two-step extractor mirrors the pipeline in our template paper
(`2412.08307`): rule-based letter/option-text match first, sentence-
transformer fallback only if rules miss. Threshold defaults to `0.5`,
override via `extractor.sim_threshold` in YAML.

For MCQ benchmarks `correct` requires the extracted letter to equal the
gold letter (case-insensitive). For open-ended benchmarks the gold string
is checked as a substring first, then via cosine similarity.

## 7. Roadmap

- [ ] Wire up additional benchmarks: SeedBench, MMMU, BLINK, TaskMeAnything.
- [ ] Add MMBench `test` split + submission-format dumper.
- [ ] Wire LLaMA-Factory training side (separate subtree `unifiedmlm/train/`).
- [ ] Multi-model batch dispatch on a single vLLM engine where possible.
- [ ] Per-category accuracy reporting (group by `category` / `l2-category`).

## 8. License

See `LICENSE`.
