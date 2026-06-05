# 环境安装指南（uv 单 venv 版）

> 2026-05 重写：原来的 2~3 个 conda env 全部收敛到**一个 uv venv**。所有 13 个
> 模型的 wrapper（含 vLLM 原生 + HF transformers backend 兜底）都在同一份依赖里
> 工作；不需要按模型切环境。
>
> 历史背景见末尾"为什么不再用 conda 多 env"。

---

## 0. 前置要求

| 项 | 实测要求 |
|---|---|
| NVIDIA driver | ≥ 550 (CUDA 12.4 forward-compat)，cu128 wheels OK |
| 显卡 | RTX 4090 24G × N。**单卡能跑的模型**：≤ 9 B fp/bf16；更大走 `device_map="auto"` 跨卡 |
| Python | 3.11 (uv 自动下载) |
| 磁盘 | venv ~10 G；HF cache 几十 G 起步，建议放共享盘 |
| 工具 | `uv`（一行装到 conda base：`pip install uv`） |

---

## 1. 装环境

```bash
# 1. 装 uv（任选一种，写入 ~/.local/bin 或 conda base）
pip install uv                                  # 推荐：装到 conda base
# 或：curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 项目目录里创建 venv
cd /path/to/UnifiedMLM
uv venv .venv --python 3.11

# 3. 装本项目（含全部依赖）
uv pip install -e .

# 4. 校验
.venv/bin/python -c "import torch, vllm, transformers; \
  print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} | vllm {vllm.__version__} | trf {transformers.__version__}')"
# 期望（2026-05 锁定）：torch 2.9.0+cu128 / vllm 0.11.2 / trf 4.57.6
```

`pyproject.toml` 里 `[tool.uv.sources]` 已配 PyTorch cu128 index；不用手动加 `--index`。

---

## 1.5 Qwen3.5 单独 venv（A100 测试机）

> Qwen3.5（原生多模态，无单独 "-VL" 线；4B / 9B / 397B-A17B）需要
> **vLLM ≥ 0.17**，主 venv 锁的 0.11.2 跑不了，也不要把主 venv 升上去
> （会破坏 §4 里靠 0.11.2 验证过的 13 个 wrapper）。单开一个 venv：

```bash
cd /path/to/UnifiedMLM && git pull

# 1. 单独 venv（Python 3.11，与主 .venv 并存）
uv venv .venv-qwen35 --python 3.11

# 2. 先装新 vLLM（自带配套 torch / transformers）。
#    --torch-backend=auto 让 uv 按 nvidia-smi 驱动版本选 cu 轮子；
#    A100 驱动较老（<580）时新 torch 的 cu13 轮子会报 driver too old，
#    那就显式 --torch-backend=cu128。
uv pip install -p .venv-qwen35 "vllm>=0.18" --torch-backend=auto

# 3. 项目本体 --no-deps 装入（绕开 pyproject 里 vllm 0.11.2 的旧 pin），
#    再补 eval 链路的轻量依赖
uv pip install -p .venv-qwen35 -e . --no-deps
uv pip install -p .venv-qwen35 datasets sentence-transformers scikit-learn \
  pyyaml tqdm pandas pillow huggingface-hub hf-transfer

# 4. 校验
.venv-qwen35/bin/python -c "import torch, vllm, transformers; \
  print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} | vllm {vllm.__version__} | trf {transformers.__version__}')"
```

下载权重 + 冒烟（HF_HOME 仍按 §2 共享）：

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 .venv-qwen35/bin/hf download Qwen/Qwen3.5-4B

CUDA_VISIBLE_DEVICES=0 \
  .venv-qwen35/bin/unifiedmlm-eval --config configs/eval/qwen3_5_4b_mmbench_subset.yaml
```

注册名：`qwen3.5-4b` / `qwen3.5-9b`（A100 80G 单卡 9B 也轻松，改 yaml 的
`model_path` 即可）。注意事项：

| 事项 | 说明 |
|---|---|
| thinking mode | Qwen3.5 默认开。config `enable_thinking: false` 走 chat template 关掉（MCQ 评测默认）；wrapper 对漏出来的 `<think>` 块有 strip 兜底，`per_sample.jsonl` 的 metadata 里 `had_think_block` 可查 |
| `Unknown model class 'Qwen3_5ForConditionalGeneration'` | 用错 venv 了（vLLM < 0.17）；确认命令走 `.venv-qwen35/bin/` |
| `driver too old` | torch 拉了 cu13 轮子；重装时 `--torch-backend=cu128` |
| TRL GRPO colocate 报 weight prefix mismatch | 已知上游问题（QwenLM/Qwen3.6#64），只影响训练 colocate，纯推理评测不受影响 |

---

## 2. HF 缓存

所有 backend（vLLM、transformers）共用同一份 `HF_HOME`。

```bash
# 写进 ~/.bashrc 或 .envrc 之类的
export HF_HOME=/mnt/sdc/zimuwang/huggingface            # 你机器上的大盘
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_ENABLE_HF_TRANSFER=1                       # 加速下载
unset TRANSFORMERS_CACHE                                 # ⚠️ 老陷阱：别保留
```

> ⚠️ **TRANSFORMERS_CACHE 陷阱**：如果以前 `~/.bashrc` 设过 `TRANSFORMERS_CACHE`
> 且跟 `HF_HOME/hub` 不一致，transformers 会优先读它，导致 vLLM 下到 HF_HOME 的
> 权重 transformers 找不到。**直接删掉那一行**。

---

## 3. 跑评测

```bash
cd /path/to/UnifiedMLM
source .venv/bin/activate                                # 或直接调 .venv/bin/unifiedmlm-eval

# 单卡 vLLM 模型（GPU 0）
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  unifiedmlm-eval --config configs/eval/qwen3vl_mmbench_subset.yaml

# 多卡 HF backend 模型（跨 GPU 2+3，device_map="auto" 自动 split）
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  unifiedmlm-eval --config configs/eval/pixtral12b_mmbench_subset.yaml
```

---

## 4. 模型 / Backend 一览（13 个 wrapper）

| 模型 | 注册名 | Backend | 卡数 (24G) | 备注 |
|---|---|---|---|---|
| LLaVA-1.5-7B | `llava-1.5-7b` | vLLM | 1 | 论文复现基线 |
| Qwen2.5-VL-7B | `qwen2.5-vl-7b` | vLLM | 1 | `limit_mm_per_prompt video=0` 必加 |
| Qwen3-VL-8B | `qwen3-vl-8b` | vLLM | 1 | 同上；`max_model_len=8192` |
| InternVL3.5-8B | `internvl3.5-8b` | vLLM | 1 | `max_dynamic_patch=6` 控视觉 token |
| Phi-4 Multimodal | `phi-4-multimodal` | vLLM | 1 | `batch_size: 1`（不同分辨率不能 batch）|
| LLaVA-OneVision-1.5-8B | `llava-onevision-1.5-8b` | **HF** | 1 | 走 `qwen_vl_utils.process_vision_info` |
| LLaVA-OneVision-2-8B | `llava-onevision-2-8b` | **HF** | 1 (80G) | 需 trf ≥ 5.7（A100 `.venv-qwen35` / A800 `.venv-ov2`）；`AutoModelForImageTextToText` + remote code，另需 `opencv-python-headless`。A800 实测 0.97 @ mmbench-subset。⚠️ A800 上 HF 下载走 `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1` 直连（mihomo 代理会静默卡死）；pypi 用清华镜像 |
| Pixtral-12B | `pixtral-12b` | **HF, `device_map=auto`** | 2 | 12 B 单卡装不下；TP=2 NCCL fail |
| Molmo2-O-7B | `molmo2-o-7b` | **HF** | 1 | `AutoModelForImageTextToText` + remote code |
| Molmo2-8B / 4B | `molmo2-8b` / `molmo2-4b` | **HF** | 1 | 同上模板（未下载权重）|
| GLM-4V-9B | `glm-4v-9b` | **HF, `device_map=auto`** | 2 | fp32 存储 cast 峰值大；`use_cache=False` + `num_hidden_layers` patch |
| Kimi-VL-A3B | `kimi-vl-a3b` | **HF, `device_map=auto`** | 2 | `PytorchGELUTanh` monkey-patch + `use_cache=False` |
| DeepSeek-VL2-Small | `deepseek-vl2-small` | (blocked — 见 §6 / wrapper docstring) | 2 | vLLM TP=2 vocab assertion + HF backend RoPE 不兼容；2 条 upstream fix path |

> wrapper 全部支持 `config.backend: vllm | hf`（HF 之外的有 vLLM-only 也接受 `backend: vllm`
> 显式）。各 yaml 默认值见 `configs/eval/*.yaml`。

### 4.5 vLLM chat-API 批次（2026-06，A800 `.venv-vllm`）

> 共享基类 `unifiedmlm/models/vllm_chat.py`：走 `LLM.chat()` 让 vLLM 自己注入
> vision placeholder（11 个家族模板各异，不再手写）；图像传 base64 data URL。
> 需要**新 vLLM**（≥0.12 区间，A800 上单独 `.venv-vllm`），主 venv 0.11.2 跑不全。

| 模型 | 注册名 | 规模 | 备注 |
|---|---|---|---|
| Gemma 3 | `gemma-3-4b` / `gemma-3-12b` | 4B/12B | ⚠️ gated，下载需 HF_TOKEN；强制 bf16 |
| MiniCPM-V 4.5 | `minicpm-v-4.5` | 8B | 混合 thinking 默认关 |
| Ovis 2.5 | `ovis2.5-9b` / `ovis2.5-2b` | 9B/2B | |
| GLM-4.1V-Thinking | `glm-4.1v-9b-thinking` | 9B | thinking 无开关，wrapper 剥 `<answer>`；可替代老 GLM-4V-9B |
| Mistral Small 3.1 | `mistral-small-3.1-24b` | 24B | 80G 单卡；HF 格式跑不通时 yaml 开 mistral 三件套 |
| Aya Vision | `aya-vision-8b` | 8B | ⚠️ gated（CC-BY-NC），需 HF_TOKEN |
| ERNIE 4.5 VL | `ernie-4.5-vl-28b-a3b` | 28B-A3B | 权重 ~56G，gpu_memory_util 0.95 |
| Keye-VL 1.5 | `keye-vl-1.5-8b` | 8B | auto-thinking，strip 兜底 |
| Eagle 2.5 | `eagle2.5-8b` | 8B | |
| SmolVLM2 | `smolvlm2-2.2b` | 2.2B | small 端点 |
| Intern-S1-mini | `intern-s1-mini` | 8B | `chat_template_kwargs: {enable_thinking: false}` |

跳过未接：Step3-VL / MiniMax-VL-01 / GLM-4.5V / ERNIE-VL-424B（百 B 级，单机放不下）、
HunyuanVL（HF repo 401，未确认开放权重）。Molmo2 / Kimi-VL / DeepSeek-VL2 的 vLLM
原生支持已进上游，主线版本到位后把各自 yaml 的 `backend` 切回 `vllm` 即可提速。

---

## 5. MMBench-subset (100q) 实测分数（GPU 0/2+3，单 uv venv）

| 模型 | acc | 耗时 |
|---|---|---|
| Qwen2.5-VL-7B | 0.90 | 7.8 s |
| Qwen3-VL-8B | 0.96 | 6.4 s |
| InternVL3.5-8B | 0.90 | 17.1 s |
| LLaVA-OV-1.5-8B | 0.99 | 10.9 s |
| LLaVA-OV-2-8B | 0.97 | 10.3 s（A800 80G 单卡，.venv-ov2） |
| Pixtral-12B | 0.89 | 17.6 s |
| Molmo2-O-7B | 0.96 | 16.5 s |
| Phi-4 Multimodal | 0.71 | 19.5 s |
| GLM-4V-9B | 1.00 | 150.7 s |
| Kimi-VL-A3B | 0.97 | 188.3 s |

GLM-4V / Kimi-VL 慢是因为 `use_cache=False`（remote code 跟新 cache API 不兼容的解法）。

---

## 6. 常见问题

| 现象 | 排查 |
|---|---|
| `Unknown model class 'Qwen3VLForConditionalGeneration'` | vLLM 装老了；`uv pip install --upgrade vllm` |
| `Model architectures ['LLaVAOneVision1_5_*'] are not supported` | 把 yaml 改成 `backend: hf` |
| `Molmo2Processor has no attribute _get_num_multimodal_tokens` | vLLM 0.11.2 不识别 Molmo2 arch → `backend: hf` |
| `Unrecognized configuration class ... for AutoModelForCausalLM` | Molmo 2 用 `AutoModelForImageTextToText`；wrapper 已处理 |
| `OSError: We couldn't connect to 'https://hf-mirror.com'... and couldn't find in cached files` | `unset TRANSFORMERS_CACHE` 后重试；并删 `~/.bashrc` 那一行 |
| `CUDA out of memory`（vLLM 加载时）| 12 B+ 模型单卡装不下 → 把 yaml 改成 `backend: hf` + `device_map: auto`，跑时 `CUDA_VISIBLE_DEVICES=多卡` |
| `KV cache memory ... too small` | 降 `max_model_len`（8192 通常够）或升 `gpu_memory_util` 到 0.95 |
| GLM-4V `'ChatGLMConfig' has no attribute 'num_hidden_layers'` | wrapper `_init_hf` 里 monkey-patch 已处理；如果手写代码遇到，加 `cfg.num_hidden_layers = cfg.num_layers` |
| Kimi-VL `cannot import name 'PytorchGELUTanh'` | wrapper `_init_hf` 已 alias 到 `GELUTanh`；如果手写代码遇到，加同样 patch |
| `driver too old (found 12040)` | torch wheel 是 cu13；用 cu128 wheel（`pyproject.toml` 默认） |
| DeepSeek-VL2 vLLM `loaded_weight.shape[output_dim] == self.org_vocab_size` | vLLM 0.11.2 加载 DeepSeek-VL2 + TP=2 时的 known bug。等 vLLM 上游修复或升级。 |
| DeepSeek-VL2 HF backend `cannot import name 'LlamaFlashAttention2'` | deepseek-vl2 库与 transformers 4.57 不兼容；要 4 个 monkey-patch + GenerationMixin 注入 + RoPE 改源码才行（不推荐）。详见 `unifiedmlm/models/deepseek_vl2.py` docstring。 |

---

## 7. 为什么不再用 conda 多 env

| 历史方案 | 实际遇到 |
|---|---|
| env-legacy: vllm 0.8.5 跑 LLaVA-1.5 + Qwen2.5-VL | vLLM 0.11.2 完全向后兼容这两个，没必要老版本 |
| env-current: vllm 0.10-0.11 跑新 vLLM 原生模型 | ✅ 保留作为核心，迁到 uv |
| env-molmo2: vllm ≥ 0.11 跑 Molmo 2 | vLLM 主线没合并 Molmo2；升 vLLM 0.21 又 cu13 不兼容驱动 → 改走 HF backend |
| 一些模型（LLaVA-OV-1.5）vLLM 完全不支持 | HF backend 兜底 |

结果：**单 vLLM 0.11.2 + HF backend 兜底 = 13 个 wrapper 都能在同一个 venv 里跑**。

---

## 8. 一句话总结

```bash
uv venv .venv --python 3.11 && uv pip install -e .
```

然后 `CUDA_VISIBLE_DEVICES=... .venv/bin/unifiedmlm-eval --config configs/eval/<x>.yaml` 即可。
