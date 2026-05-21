# UnifiedMLM

Unified evaluation (and, later, training) pipeline for multimodal LLMs.
The evaluation side is vLLM-accelerated and registry-based so new models and
benchmarks plug in without touching the runner.

This repo is the engineering backbone for the template-scaling work in
`2412.08307` and its extensions — it standardizes the eval口径 so every
experiment (Scaling-Law, Test-Train Distance, Local Competition, …) reports
numbers that are directly comparable.

> Current status: **v0.2.0** — vLLM + HF-transformers hybrid eval line, **13 models
> wired up** in a **single uv venv** (LLaVA-1.5-7B, Qwen2.5-VL-7B, Qwen3-VL-8B,
> InternVL3.5-8B, LLaVA-OneVision-1.5-8B, Pixtral-12B, Molmo2-8B/O-7B/4B,
> Phi-4-Multimodal, GLM-4V-9B, Kimi-VL-A3B, DeepSeek-VL2-Small).
> One benchmark wired (MMBench-en/dev). Training (LLaMA-Factory) lands later.

## ⚡ 环境与模型一览（先看这里）

**单 uv venv 通吃所有 13 个模型**：vLLM 0.11.2 + transformers 4.57.6 + torch
2.9+cu128。能 vLLM 原生支持的走 vLLM，新架构 vLLM 还没合并的（LLaVA-OV-1.5、
Molmo2、Pixtral 大模型等）走 HF transformers backend 兜底（wrapper 里 `backend: hf`
一行就切）。完整安装步骤见 [`docs/ENVS.md`](docs/ENVS.md)。

```bash
uv venv .venv --python 3.11 && uv pip install -e .
```

### Backend / 卡数

| 模型 | 注册名 | Backend | 单卡 24G | MMBench-subset acc |
|---|---|---|---|---|
| LLaVA-1.5-7B | `llava-1.5-7b` | vLLM | 1 | — |
| Qwen2.5-VL-7B | `qwen2.5-vl-7b` | vLLM | 1 | 0.90 |
| Qwen3-VL-8B | `qwen3-vl-8b` | vLLM | 1 | 0.96 |
| InternVL3.5-8B | `internvl3.5-8b` | vLLM | 1 | 0.90 |
| Phi-4 Multimodal | `phi-4-multimodal` | vLLM | 1 | 0.71 |
| LLaVA-OV-1.5-8B | `llava-onevision-1.5-8b` | HF | 1 | 0.99 |
| Molmo2-O-7B | `molmo2-o-7b` | HF | 1 | 0.96 |
| Pixtral-12B | `pixtral-12b` | HF, `device_map=auto` | **2** | 0.89 |
| GLM-4V-9B | `glm-4v-9b` | HF, `device_map=auto` | **2** | 1.00 |
| Kimi-VL-A3B | `kimi-vl-a3b` | HF, `device_map=auto` | **2** | 0.97 |
| DeepSeek-VL2-Small | `deepseek-vl2-small` | vLLM TP=2 (待 fix) | 2 | (vLLM 0.11.2 bug) |
| Molmo2-8B / 4B | `molmo2-8b` / `molmo2-4b` | HF | 1 | (未下载权重) |

> 模型权重通过共享 `HF_HOME` 缓存，cross-backend / cross-model 复用。

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

## 2. Install (uv venv, single environment)

```bash
# 1. 装 uv（任选一种）
pip install uv                                  # 推荐：装到 conda base 或系统 python
# 或：curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 创建 venv + 装本项目（含 vLLM 0.11.2 + torch 2.9+cu128 + transformers 4.57）
uv venv .venv --python 3.11
uv pip install -e .

# 3. 验证
.venv/bin/python -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"
.venv/bin/unifiedmlm-eval --list
```

`pyproject.toml` 里 `[tool.uv.sources]` 已经把 PyTorch index 指到 cu128 build
（forward-compatible with CUDA 12.4 driver）；不需要手工指定 `--index`。

详见 [`docs/ENVS.md`](docs/ENVS.md)。

## 3. Smoke test (100 samples)

```bash
source .venv/bin/activate                       # 或 .venv/bin/unifiedmlm-eval

# 单卡 vLLM 模型
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  unifiedmlm-eval --config configs/eval/qwen3vl_mmbench_subset.yaml

# 多卡 HF backend 模型（跨 GPU 2+3）
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  unifiedmlm-eval --config configs/eval/pixtral12b_mmbench_subset.yaml

# 或 ad-hoc 不带 YAML
unifiedmlm-eval --model llava-1.5-7b --benchmark mmbench --limit 100 \
  --batch-size 16 --output-dir outputs/smoke_llava15_mmbench
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
