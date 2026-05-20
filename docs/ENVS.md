# 多环境安装指南

> 为什么分多个 env：新模型对 vLLM / transformers / torch 版本的最低要求不同，硬塞一个 env 容易冲突。
>
> **2026-05 实测后简化**：把原计划的 3 个 env 缩到 **2 个**（env-legacy + env-current）。
> 原 env-C (molmo2) 因为驱动锁 CUDA 12.4 而走不通（详见末尾"踩坑记录"），改用
> **HF transformers backend** 作为兜底，所以 env-current 一个就能跑全部 5 个新模型。

---

## 总览：每个模型对应哪个 env / backend

| 模型 | 注册名 | env | backend | 备注 |
|---|---|---|---|---|
| LLaVA-1.5-7B | `llava-1.5-7b` | **env-A (legacy)** | vLLM 0.8.5 | 论文复现基线 |
| Qwen2.5-VL-7B-Instruct | `qwen2.5-vl-7b` | **env-A (legacy)** | vLLM 0.8.5 | v1.1 cross-arch |
| Qwen3-VL-8B-Instruct | `qwen3-vl-8b` | **env-B (current)** | vLLM 0.11.2 | Qwen3-VL 原生支持 |
| InternVL3.5-8B | `internvl3.5-8b` | **env-B (current)** | vLLM 0.11.2 | InternVLChatModel 原生支持 |
| LLaVA-OneVision-1.5-8B | `llava-onevision-1.5-8b` | **env-B (current)** | **HF transformers** | vLLM 还没合并新架构，走 `backend: hf` |
| Molmo2-O-7B | `molmo2-o-7b` | **env-B (current)** | **HF transformers** | 同上，走 `backend: hf` |
| Molmo2-8B / 4B | `molmo2-8b` / `molmo2-4b` | **env-B (current)** | **HF transformers** | 同 Molmo2-O-7B 模式（未实测，需下载权重） |

**取舍**：2 个 env 占磁盘约 2 × 8 GB（conda env） + 一份共享的 HF 缓存（30–80 GB，看下了几个模型）。

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

## Env B：current（vLLM 新版 + HF backend 兜底，跑 5 个新模型）

支持模型（vLLM 原生）：`qwen3-vl-8b`, `internvl3.5-8b`
支持模型（HF transformers backend）：`llava-onevision-1.5-8b`, `molmo2-o-7b`, `molmo2-8b`, `molmo2-4b`

为什么混 backend：vLLM 0.11.2 已经识别 Qwen3-VL/InternVL3.5 的 arch，但 LLaVA-OV-1.5 的
`LLaVAOneVision1_5_ForConditionalGeneration`（2025-09 新架构）和 Molmo 2 的
`Molmo2ForConditionalGeneration`（2025-12 新架构）截至 2026-05 都还没进 vLLM 主线。
试过升 vLLM 到 0.21.0（已合并 Molmo 2）— 它的 cu13 wheel 在 CUDA-12.4 驱动上跑不起来。
所以这两个模型走 transformers backend 兜底。

```bash
conda create -n unifiedmlm-current python=3.11 -y
conda activate unifiedmlm-current

pip install --upgrade pip

# 注意：cu124 channel 上 torch 最高只到 2.6.0；让 vLLM 自己拉合适版本（会装到 cu128 wheel，
# 在 driver 12.4 上跨次版本兼容）。
pip install "vllm>=0.10.0,<0.12.0"

# HF backend 依赖：LLaVA-OV-1.5 需要 qwen-vl-utils 处理 vision input。
# 其他常规依赖。
pip install qwen-vl-utils \
            sentence-transformers \
            datasets \
            hf-transfer \
            accelerate \
            pyyaml pillow numpy pandas einops

cd /path/to/UnifiedMLM
pip install -e .
```

**实测装出来的版本组合（2026-05）**：
- torch 2.9.0+cu128
- vllm 0.11.2
- transformers 4.57.6
- driver 12.4 上 cu128 wheel forward-compatible，跑通

**校验**（GPU 0，MMBench-subset 100 题）：
```bash
python -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"

# vLLM 原生
unifiedmlm-eval --config configs/eval/qwen3vl_mmbench_subset.yaml      # acc=0.96, 6.4s
unifiedmlm-eval --config configs/eval/internvl3_5_mmbench_subset.yaml  # acc=0.90, 17.1s

# HF backend（YAML 里 backend: hf 已经预设）
unifiedmlm-eval --config configs/eval/llava_ov15_mmbench_subset.yaml   # acc=0.99, 10.9s
unifiedmlm-eval --config configs/eval/molmo2_o_7b_mmbench_subset.yaml  # acc=0.96, 16.5s
```

**已知潜在坑**：
- InternVL3.5 的 `max_dynamic_patch` 默认 12 会产生 ~3000 visual tokens；OOM 时降到 6
- Qwen3-VL 的 `max_pixels` 默认很高；OOM 时降到 `1003520`（约 1280×784）
- vLLM 给 Qwen-VL family profiling 时会预留 video encoder cache（4 dummy video）→ 24G 卡
  KV cache OOM。`qwen2_5_vl.py` / `qwen3_vl.py` 里 `limit_mm_per_prompt={"image": 1, "video": 0}`
  显式禁掉
- HF backend 必须 `dtype: bfloat16` 加载，fp32 撑爆 24G 卡（Molmo2-O-7B fp32 = 28G）

---

## HF transformers backend 说明（v0.1.1 新增）

两个 wrapper（`llava_onevision15.py` / `molmo2.py`）现在支持 `backend` config key：

```yaml
# Molmo2-O-7B（默认走 HF）
model:
  name: molmo2-o-7b
  config:
    backend: hf            # "vllm" | "hf"
    model_path: allenai/Molmo2-O-7B
    dtype: bfloat16        # 24G 卡必须
    device: cuda:0
    sampling:
      temperature: 0.0
      max_tokens: 128

# LLaVA-OV-1.5（默认走 HF）
model:
  name: llava-onevision-1.5-8b
  config:
    backend: hf
    model_path: lmms-lab/LLaVA-OneVision-1.5-8B-Instruct
    dtype: bfloat16
    device: cuda:0
    sampling:
      temperature: 0.0
      max_tokens: 128
```

实现细节：
- LLaVA-OV-1.5 走 `AutoModelForCausalLM` + `qwen_vl_utils.process_vision_info`（HF 卡描述
  里推荐的方式），processor 自动加载 `Qwen2_5_VLProcessor`
- Molmo 2 走 `AutoModelForImageTextToText`（auto_map 里就是这个 Auto class）；
  `Molmo2Processor` 的 batch 支持参差不齐，所以逐条 generate
- 未来 vLLM 主线合并这两个架构后，把 YAML 的 `backend` 改成 `vllm` 即可秒切

---

## 踩坑记录（为啥放弃 env-molmo2）

| 尝试 | 结果 |
|---|---|
| vLLM 0.11.2 + Molmo2 (fallback TransformersMultiModal) | `Molmo2Processor` 缺 `_get_num_multimodal_tokens` API |
| vLLM 0.12.0 + Molmo2 | arch `Molmo2ForConditionalGeneration` 还没合并 |
| vLLM 0.16.0 + Molmo2 | arch 已合并；`import vllm.distributed.kv_transfer` 包初始化时 SIGSEGV（C++ 层，faulthandler 抓不到栈）。怀疑 cu128 wheel 在 driver 12.4 上的边界 bug |
| vLLM 0.21.0 + Molmo2 | arch 完美识别；但带 torch 2.11+cu130 → "driver too old (found 12040)" |
| **HF transformers backend** | **✅ 通过，acc=0.96** |

驱动锁死 CUDA 12.4 时，没有现成 vLLM 版本能同时满足"Molmo 2 原生支持 + cu12 torch wheel"。
等 vLLM 出 cu12 wheel 的 Molmo 2 支持，或机器驱动升级，再切回 `backend: vllm`。

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
# 用法：ummm legacy / ummm current
```

> ⚠️ **TRANSFORMERS_CACHE 陷阱**：如果 `~/.bashrc` 里有 `export TRANSFORMERS_CACHE=...`
> 且指向跟 `HF_HOME/hub` 不同的目录，transformers 会优先用前者，找不到 vLLM 下到 `HF_HOME`
> 的模型权重。建议删掉这一行，只保留 `HF_HOME`。临时绕过：运行 eval 前 `unset TRANSFORMERS_CACHE`。

---

## 版本依赖矩阵（速查，2026-05 实测）

| Env | Python | torch | vLLM | transformers | 支持模型 |
|---|---|---|---|---|---|
| legacy | 3.11 | 2.6.0+cu124 | 0.8.5 | 4.51.3 | LLaVA-1.5, Qwen2.5-VL |
| current | 3.11 | 2.9.0+cu128 | 0.11.2 | 4.57.6 | Qwen3-VL, InternVL3.5（vLLM）+ LLaVA-OV-1.5, Molmo2-*（HF backend）|

---

## 常见问题

| 现象 | 排查 |
|---|---|
| `Unknown model class 'Qwen3VLForConditionalGeneration'` | env 用错了——这个 arch 只有 env-current 的 vLLM 认识 |
| `Model architectures ['LLaVAOneVision1_5_ForConditionalGeneration'] are not supported` | YAML 里加 `backend: hf`（vLLM 还没合并这个 arch）|
| `Model architectures ['Molmo2ForConditionalGeneration'] are not supported` | 同上，`backend: hf` |
| `Unrecognized configuration class ... for AutoModelForCausalLM` | Molmo 2 走 `AutoModelForImageTextToText`，wrapper 已经处理；如果是自己 import，注意用对 Auto class |
| `OSError: We couldn't connect to 'https://hf-mirror.com' ... and couldn't find them in the cached files` | 你 `~/.bashrc` 里 TRANSFORMERS_CACHE 指错了。`unset TRANSFORMERS_CACHE` 后重试 |
| `CUDA out of memory` | 降 `gpu_memory_util` → 0.85；降 batch_size；新模型同时降 `max_pixels` 或 `max_dynamic_patch`；HF backend 确保 `dtype: bfloat16` |
| `KV cache memory ... is larger than available` (Qwen-VL) | 已修：wrapper 显式 `limit_mm_per_prompt={"image": 1, "video": 0}` 禁掉 video profiling |
| `driver too old (found 12040)` | torch wheel 是 cu13 build，跟 driver 12.4 不兼容。换 cu12 wheel 的 vLLM（≤0.16） |
| 两个 env 装下来磁盘紧 | 共享 HF_HOME 是关键；conda env 本身 ~8 GB 一个，可接受 |

---

## 一句话总结

**2 个 env，共享一个 HF 缓存**：
- `unifiedmlm-legacy` ← LLaVA-1.5 + Qwen2.5-VL（论文复现）
- `unifiedmlm-current` ← 其它 5 个新模型（vLLM 原生 2 个 + HF backend 3 个 family）

切换用 `conda activate unifiedmlm-<name>`，剩下的 UnifiedMLM 代码和 config 完全通用。
驱动升级到支持 CUDA 13 后，可以把 HF backend 的两个模型切回 `backend: vllm`。
