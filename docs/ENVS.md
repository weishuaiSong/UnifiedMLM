# 多环境安装指南

> 为什么分多个 env：新模型对 vLLM / transformers / torch 版本的最低要求不同，硬塞一个 env 容易冲突。下面 3 个 env 覆盖全部 8 个注册模型；CUDA 假设 12.4（驱动支持上限），其它 CUDA 版本参考 `SETUP.md §10` 选 wheel。

---

## 总览：每个模型对应哪个 env

| 模型 | 注册名 | env | 备注 |
|---|---|---|---|
| LLaVA-1.5-7B | `llava-1.5-7b` | **env-A (legacy)** | 论文复现基线 |
| Qwen2.5-VL-7B-Instruct | `qwen2.5-vl-7b` | **env-A (legacy)** | v1.1 cross-arch |
| LLaVA-OneVision-1.5-8B | `llava-onevision-1.5-8b` | **env-B (current)** | Qwen3 backbone + SigLIP 2 |
| Qwen3-VL-8B-Instruct | `qwen3-vl-8b` | **env-B (current)** | Qwen3-VL 需要 vLLM ≥ 0.10 |
| InternVL3.5-8B | `internvl3.5-8b` | **env-B (current)** | 需要 vLLM 支持 InternViT 动态分块 |
| Molmo2-8B | `molmo2-8b` | **env-C (molmo2)** | Ai2 自定义 processor |
| Molmo2-O-7B | `molmo2-o-7b` | **env-C (molmo2)** | 同上 |
| Molmo2-4B | `molmo2-4b` | **env-C (molmo2)** | 同上 |

**取舍**：3 个 env 占磁盘约 3 × 8 GB（conda env） + 一份共享的 HF 缓存（30–80 GB，看下了几个模型）。

---

## 通用前置（每个 env 都要做一次）

```bash
# 0.1 mirror & 缓存（写进 ~/.bashrc 或每次 source）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf_cache              # 改成你机器上的大盘
export HF_HUB_ENABLE_HF_TRANSFER=1
```

```bash
# 0.2 检查 CUDA driver 上限
nvidia-smi | head -5     # 看 "CUDA Version: 12.x"
```

下面所有 env 都用 Python 3.11、CUDA 12.4 的 wheel 作为示例。CUDA 12.6/12.8 把 `cu124` 换成 `cu126` / `cu128` 即可。

---

## Env A：legacy（已有，跑 LLaVA-1.5 + Qwen2.5-VL）

支持模型：`llava-1.5-7b`, `qwen2.5-vl-7b`

```bash
conda create -n unifiedmlm-legacy python=3.11 -y
conda activate unifiedmlm-legacy

pip install --upgrade pip

# torch + vLLM（CUDA 12.4 适配版本，与现 SETUP.md 一致）
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.5

# 框架依赖
pip install transformers==4.49.0 \
            accelerate==1.2.0 \
            sentence-transformers==3.2.1 \
            datasets==3.1.0 \
            hf-transfer==0.1.8 \
            pyyaml pillow numpy pandas

# 装 UnifiedMLM 本体
cd /path/to/UnifiedMLM
pip install -e .
```

**校验**：
```bash
python -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"
# 期望：0.8.5  4.49.0
unifiedmlm-eval --config configs/eval/llava15_mmbench_subset.yaml
```

---

## Env B：current（新一代 vLLM，跑 LLaVA-OV-1.5 / Qwen3-VL / InternVL3.5）

支持模型：`llava-onevision-1.5-8b`, `qwen3-vl-8b`, `internvl3.5-8b`

这三个模型都需要较新的 vLLM 才能识别它们的 model arch。

```bash
conda create -n unifiedmlm-current python=3.11 -y
conda activate unifiedmlm-current

pip install --upgrade pip

# vLLM ≥ 0.10 才支持 Qwen3-VL 和 InternVL3.5
pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install "vllm>=0.10.0,<0.12.0"

# transformers 需要 ≥ 4.53 才识别 Qwen3-VL / InternVL3.5 的 config
pip install "transformers>=4.53.0" \
            accelerate \
            sentence-transformers \
            datasets \
            hf-transfer \
            pyyaml pillow numpy pandas

cd /path/to/UnifiedMLM
pip install -e .
```

**校验**：
```bash
python -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"
# 期望：vllm 0.10.x 或 0.11.x；transformers 4.53.x+

# 三个新模型各跑一遍 100 题 sanity
unifiedmlm-eval --config configs/eval/llava_ov15_mmbench_subset.yaml
unifiedmlm-eval --config configs/eval/qwen3vl_mmbench_subset.yaml
unifiedmlm-eval --config configs/eval/internvl3_5_mmbench_subset.yaml
```

**已知潜在坑**：
- InternVL3.5 的 `max_dynamic_patch` 默认 12 会产生 ~3000 visual tokens；OOM 时降到 6
- Qwen3-VL 的 `max_pixels` 默认很高（参考 Qwen2.5-VL 一节）；OOM 时降到 ~1003520

---

## Env C：molmo2（Ai2 Molmo 2 系列，2025-12 发布）

支持模型：`molmo2-8b`, `molmo2-o-7b`, `molmo2-4b`

Molmo 系列在 vLLM 上的支持长期跟 transformers 节奏不一致。2026-05 时 vLLM 已经 merge 了 Molmo 2 PR，但版本要求比 env-B 还激进；同时 Molmo 自带 custom processor，需要 `trust_remote_code=True`。

```bash
conda create -n unifiedmlm-molmo2 python=3.11 -y
conda activate unifiedmlm-molmo2

pip install --upgrade pip

# Molmo 2 需要更新的 vLLM；用最新 release（或 nightly 如有需要）
pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install "vllm>=0.11.0"

# Molmo 2 processor 需要 transformers 4.54+
pip install "transformers>=4.54.0" \
            "tokenizers>=0.21.0" \
            accelerate \
            sentence-transformers \
            datasets \
            hf-transfer \
            einops \
            pyyaml pillow numpy pandas

cd /path/to/UnifiedMLM
pip install -e .
```

**校验**：
```bash
python -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"
# 期望：vllm 0.11.x+；transformers 4.54.x+

unifiedmlm-eval --config configs/eval/molmo2_8b_mmbench_subset.yaml
```

**关于 `hf_overrides`**：vLLM 当前 release 加载 Molmo 2 时需要显式告诉它架构名。**这个 default 已经包在 `molmo2.py` 里了**（`DEFAULT_HF_OVERRIDES`），开箱即用，不用动。

若未来 vLLM 原生识别 Molmo 2，可在 YAML 显式关掉 default：
```yaml
model:
  name: molmo2-8b
  config:
    hf_overrides: null    # 关掉 default override
```

或者覆盖成自定义值：
```yaml
model:
  config:
    hf_overrides:
      architectures: ["Molmo2ForConditionalGeneration"]
      is_mm_prefix_lm: false
```

所有 8 个 wrapper 都支持读 `hf_overrides` config key，Molmo 2 之外的模型默认不传（保持原行为）。

**fallback**：如果 vLLM 跑不起来 Molmo 2，临时用 transformers backend：
```bash
# 加装 transformers 推理依赖
pip install bitsandbytes  # 可选量化
# 然后改 UnifiedMLM 里 molmo2.py 用 transformers.AutoModelForCausalLM 走 HF generate
# （TODO：当前 wrapper 是 vLLM 专用；HF backend 是 v0.2 工作）
```

**已知坑**：
- Molmo2-O-7B 的 chat template 跟 Qwen3 backbone 不同（用 `<|user|>` / `<|assistant|>`）；`molmo2.py` 已经分开处理
- Olmo 词表与 Qwen 完全不同，tokenizer 不能复用

---

## 多 env 协作（共享缓存 + 切换）

### 共享 HF 模型缓存

3 个 env 共用同一份 `HF_HOME=/data/hf_cache`，模型权重下载一次三个 env 都能用。**强烈推荐**，因为模型权重最大头。

```bash
# 在任意一个 env 里把 5 个新模型下到本地（一次性）
huggingface-cli download lmms-lab/LLaVA-OneVision-1.5-8B-Instruct
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
huggingface-cli download OpenGVLab/InternVL3_5-8B
huggingface-cli download allenai/Molmo2-8B
huggingface-cli download allenai/Molmo2-O-7B
huggingface-cli download allenai/Molmo2-4B
```

### 共享 UnifiedMLM 源码

`pip install -e .` 让 3 个 env 都引用同一份代码。改 wrapper 时不用每个 env 重装。

### 切环境的标准节奏

```bash
# 每次开终端
conda deactivate
conda activate unifiedmlm-current   # 或 -legacy / -molmo2
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /path/to/UnifiedMLM
```

可以写个 shell 函数：
```bash
# ~/.bashrc
ummm() {
  conda activate "unifiedmlm-$1"
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/data/hf_cache
  export HF_HUB_ENABLE_HF_TRANSFER=1
  cd ~/UnifiedMLM
}
# 用法：ummm legacy / ummm current / ummm molmo2
```

---

## 版本依赖矩阵（速查）

| Env | Python | torch | vLLM | transformers | 支持模型 |
|---|---|---|---|---|---|
| legacy | 3.11 | 2.6.0 cu124 | 0.8.5 | 4.49.0 | LLaVA-1.5, Qwen2.5-VL |
| current | 3.11 | 2.7.0 cu124 | ≥0.10,<0.12 | ≥4.53 | LLaVA-OV-1.5, Qwen3-VL, InternVL3.5 |
| molmo2 | 3.11 | 2.7.0 cu124 | ≥0.11 | ≥4.54 | Molmo2-8B/7B-Olmo/4B |

---

## 常见问题

| 现象 | 排查 |
|---|---|
| `Unknown model class 'Qwen3VLForConditionalGeneration'` | env 用错了——这个 arch 只有 env-B 的 vLLM 认识 |
| `Unknown model class 'Molmo2ForCausalLM'` | 切到 env-molmo2 |
| `transformers` 不识别新 config | 升级 transformers 到对应 env 的最低版 |
| `CUDA out of memory` | 降 `gpu_memory_util` → 0.85；降 batch_size；新模型同时降 `max_pixels` 或 `max_dynamic_patch` |
| Molmo 2 起不来 | vLLM 版本可能太老；先试 `pip install -U vllm`，仍不行则参考 fallback 一节走 transformers backend |
| 三个 env 装下来磁盘紧 | 共享 HF_HOME 是关键；conda env 本身 ~8 GB 一个，可接受 |

---

## 一句话总结

**装 3 个 env，共享一个 HF 缓存**：
- `unifiedmlm-legacy` ← LLaVA-1.5 + Qwen2.5-VL（论文复现）
- `unifiedmlm-current` ← LLaVA-OV-1.5 + Qwen3-VL + InternVL3.5（v1.2 主阵容）
- `unifiedmlm-molmo2` ← Molmo 2 全家（cross-arch 关键）

切换用 `conda activate unifiedmlm-<name>`，剩下的 UnifiedMLM 代码和 config 完全通用。
