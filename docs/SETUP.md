# 环境安装与运行指南

> 目标：在 Linux + NVIDIA GPU 机器上把 UnifiedMLM 装起来，跑通 LLaVA-1.5-7B 在 MMBench 上的评测。

## 0. 前置硬件 / 系统要求

| 项 | 最低 | 建议 |
|---|---|---|
| OS | Ubuntu 20.04 / 22.04 (vLLM 不支持原生 Windows，Windows 用户走 WSL2 或远端 Linux GPU 机器) | 22.04 |
| GPU | 1× 24GB（LLaVA-1.5-7B fp16 需 ~14GB 权重 + KV cache）| 1× A100-40G / A6000 / 4090 |
| CUDA | 12.1+ | 12.4 |
| Python | 3.10 | 3.11 |
| 磁盘 | 50GB（模型 ~14GB + HF 数据集缓存 + 输出）| 100GB+ |

确认 GPU 与 driver：
```bash
nvidia-smi
```

## 1. 拉代码

```bash
git clone https://github.com/weishuaiSong/UnifiedMLM.git
cd UnifiedMLM
```

## 2. 建隔离环境

二选一。推荐 conda。

### 2a. conda
```bash
conda create -n unifiedmlm python=3.11 -y
conda activate unifiedmlm
```

### 2b. venv
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## 3. 装依赖

vLLM 会自带 torch wheel；**不要**先单独装 torch，直接装 vLLM 即可：

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

校验：
```bash
python -c "import vllm; print(vllm.__version__)"
python -c "from unifiedmlm.models import list_models; from unifiedmlm.benchmarks import list_benchmarks; print('models:', list_models()); print('benchmarks:', list_benchmarks())"
```
应输出：
```
models: ['llava-1.5-7b']
benchmarks: ['mmbench']
```

> **CUDA 版本不匹配？** 如果默认 vLLM wheel 与你机器 CUDA 对不上，按 https://docs.vllm.ai/en/latest/getting_started/installation.html 选对应 wheel，再 `pip install -r requirements.txt --no-deps` 跳过 torch 重装。

## 4. HuggingFace 配置

LLaVA-1.5-7B 权重 (~14GB) 和 MMBench 都从 HF 拉。如果机器在国内访问慢：

```bash
# 用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 缓存盘建议指到大盘
export HF_HOME=/data/hf_cache         # 或你自己的大盘路径
export HF_HUB_ENABLE_HF_TRANSFER=1    # 大文件加速（需 pip install hf-transfer）

# 若需要私有/门控数据集
huggingface-cli login
```

**预下模型**（可选，提前下好避免首次跑时阻塞）：
```bash
huggingface-cli download llava-hf/llava-1.5-7b-hf
huggingface-cli download lmms-lab/MMBench --repo-type dataset
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
```

## 5. 冒烟测试（50 题，~5–10 分钟）

```bash
unifiedmlm-eval \
  --model llava-1.5-7b \
  --benchmark mmbench \
  --limit 50 \
  --batch-size 8 \
  --output-dir outputs/smoke_llava15_mmbench
```

预期输出末尾：
```
[llava-1.5-7b @ mmbench] accuracy = 0.xxxx  (xx/50, extracted=xx)  elapsed=xx.xs
```
产出：
- `outputs/smoke_llava15_mmbench/summary.json` — 总览
- `outputs/smoke_llava15_mmbench/per_sample.jsonl` — 每题 `gold / raw / predicted / extraction_method / correct`

如果冒烟通过，再上正式 run。

## 6. 正式 run（MMBench dev 全量，~4376 题）

```bash
unifiedmlm-eval --config configs/eval/llava15_mmbench.yaml
```

需要改超参就直接编辑 `configs/eval/llava15_mmbench.yaml`：
- `model.config.tensor_parallel`：多卡时调
- `model.config.gpu_memory_util`：显存吃紧时降到 0.85
- `model.config.sampling.temperature`：默认 0（贪心，复现性强）
- `benchmark.config.split`：`dev`（带 label，可本地打分）/ `test`（要提交，需自己接 dumper）
- `extractor.sim_threshold`：默认 0.5，可调

## 7. 常见问题排查

| 症状 | 排查 |
|---|---|
| `CUDA out of memory` | 降 `gpu_memory_util` 到 0.85；降 `max_model_len` 到 2048；降 `batch_size` |
| 模型死循环吐 `</s></s>...` | 检查 `sampling.max_tokens`，MCQ 任务 64–128 足够 |
| MMBench 读不到 image | HF 镜像问题，重试 / 用 `HF_HUB_ENABLE_HF_TRANSFER=1` |
| `Unknown model 'xxx'` | 该模型还没注册，看 §README "Extending" |
| sentence-transformer 下载慢 | `huggingface-cli download sentence-transformers/all-MiniLM-L6-v2` 预下 |
| 抽取 `method=="none"` 占比高 | 模型输出格式异常，查 `per_sample.jsonl` 的 `raw` 字段，可能是 prompt 模板不对或采样温度过高 |

## 8. 一键脚本（可选）

把上面流程封成脚本：

```bash
# scripts/run_smoke.sh
#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
unifiedmlm-eval \
  --model llava-1.5-7b \
  --benchmark mmbench \
  --limit 50 \
  --batch-size 8 \
  --output-dir outputs/smoke_$(date +%Y%m%d_%H%M%S)
```

```bash
chmod +x scripts/run_smoke.sh
./scripts/run_smoke.sh
```
