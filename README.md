# UnifiedMLM

Unified evaluation (and, later, training) pipeline for multimodal LLMs.
The evaluation side is vLLM-accelerated and registry-based so new models and
benchmarks plug in without touching the runner.

This repo is the engineering backbone for the template-scaling work in
`2412.08307` and its extensions — it standardizes the eval口径 so every
experiment (Scaling-Law, Test-Train Distance, Local Competition, …) reports
numbers that are directly comparable.

> Current status: **v0.1.0** — vLLM eval line, 8 models wired up
> (LLaVA-1.5-7B, Qwen2.5-VL-7B, LLaVA-OneVision-1.5-8B, Qwen3-VL-8B,
> Molmo2-8B, Molmo2-O-7B, Molmo2-4B, InternVL3.5-8B), one benchmark
> wired up (MMBench-en/dev). Training (LLaMA-Factory) will land in a
> sibling subtree later.

## ⚡ 环境与模型一览（先看这里）

所有 8 个模型都跑在 **vLLM** 上，但**不同模型对 vLLM 版本要求不同**——硬塞一个 env 会冲突。建议按下表装 **3 个独立 conda env**（共享 HF 缓存），完整安装步骤见 [`docs/ENVS.md`](docs/ENVS.md)。

| 模型 | 注册名 | vLLM 支持 | 推荐 env |
|---|---|---|---|
| LLaVA-1.5-7B | `llava-1.5-7b` | ✅ 稳定（≥0.4）| `unifiedmlm-legacy` |
| Qwen2.5-VL-7B-Instruct | `qwen2.5-vl-7b` | ✅ 稳定（≥0.7）| `unifiedmlm-legacy` |
| LLaVA-OneVision-1.5-8B | `llava-onevision-1.5-8b` | ✅ 需 vLLM ≥ 0.10 | `unifiedmlm-current` |
| Qwen3-VL-8B-Instruct | `qwen3-vl-8b` | ✅ 需 vLLM ≥ 0.10 | `unifiedmlm-current` |
| InternVL3.5-8B | `internvl3.5-8b` | ✅ 需 vLLM ≥ 0.10 | `unifiedmlm-current` |
| Molmo2-8B | `molmo2-8b` | ⚠️ 需 vLLM ≥ 0.11 + `--hf-overrides '{"architectures":["Molmo2ForConditionalGeneration"]}'` | `unifiedmlm-molmo2` |
| Molmo2-O-7B | `molmo2-o-7b` | ⚠️ 同上 | `unifiedmlm-molmo2` |
| Molmo2-4B | `molmo2-4b` | ⚠️ 同上 | `unifiedmlm-molmo2` |

**三个 env 的速查**（详细 pip 命令见 `docs/ENVS.md`）：

| Env | torch | vLLM | transformers | 占盘 |
|---|---|---|---|---|
| `unifiedmlm-legacy` | 2.6.0 cu124 | 0.8.5 | 4.49.0 | ~8 GB |
| `unifiedmlm-current` | 2.7.0 cu124 | ≥0.10,<0.12 | ≥4.53 | ~8 GB |
| `unifiedmlm-molmo2` | 2.7.0 cu124 | ≥0.11 | ≥4.54 | ~8 GB |

> 模型权重通过共享 `HF_HOME` 跨 env 共用，**不需要下三遍**。
> CUDA 12.4 / 12.6 / 12.8 切换办法见 `docs/SETUP.md §10`。

---

## 1. Layout

```
UnifiedMLM/
├── unifiedmlm/
│   ├── models/             # model wrappers (BaseVLMModel + registry)
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── llava.py                  # LLaVA-1.5-7B
│   │   ├── qwen2_5_vl.py             # Qwen2.5-VL-7B-Instruct
│   │   ├── llava_onevision15.py      # LLaVA-OneVision-1.5-8B (2025-09)
│   │   ├── qwen3_vl.py               # Qwen3-VL-8B-Instruct (2025)
│   │   ├── molmo2.py                 # Molmo2-8B / O-7B / 4B (Ai2, 2025-12)
│   │   └── internvl3_5.py            # InternVL3.5-8B (2025-08)
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
│       ├── llava15_mmbench.yaml
│       ├── llava15_mmbench_subset.yaml
│       ├── qwen25vl_mmbench_subset.yaml
│       ├── llava_ov15_mmbench_subset.yaml
│       ├── qwen3vl_mmbench_subset.yaml
│       ├── molmo2_8b_mmbench_subset.yaml
│       ├── molmo2_o_7b_mmbench_subset.yaml
│       └── internvl3_5_mmbench_subset.yaml
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
