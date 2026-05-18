# 环境安装与运行指南

> 目标：在 Linux + NVIDIA GPU 机器上把 UnifiedMLM 装起来，跑通 LLaVA-1.5-7B 在 `shijianS01/mmbench-subset`（100 题子集）上的评测。

## 0. 前置检查

| 项 | 要求 |
|---|---|
| OS | Linux（Ubuntu 20.04 / 22.04）。vLLM 不支持原生 Windows，Windows 用 WSL2 或远端机器 |
| GPU | ≥ 1× 24GB 显存（A100 / A6000 / 4090 / 3090 均可） |
| CUDA driver | ≥ 12.1 |
| Python | 3.10 或 3.11（**不要 3.8 / 3.9**，vLLM 与 numpy ≥1.26 都不支持） |
| 磁盘 | ≥ 50GB（模型权重 ~14GB + HF 缓存） |
| conda | miniconda 或 anaconda，已 `conda init` |

```bash
nvidia-smi          # 看 GPU 与 driver
conda --version     # 看 conda 是否可用
```

## 1. 拉代码

```bash
git clone https://github.com/weishuaiSong/UnifiedMLM.git
cd UnifiedMLM
```

## 2. 建 conda 环境（Python 3.11）

```bash
conda create -n unifiedmlm python=3.11 -y
conda activate unifiedmlm
```

**关键校验**：激活后必须看到 3.11，否则后续会重复踩 numpy 装不上的坑。

```bash
which python           # 应指向 ~/miniconda3/envs/unifiedmlm/bin/python
python --version       # 必须是 3.11.x
```

> 注意：conda env 内**只用 `python` 和 `pip`**（不带 3）。`python3` 仍指向系统 /usr/bin/python3.8，用它就全错。

## 3. 装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

vLLM 会自带匹配的 torch wheel，不用单独装 torch。

校验：
```bash
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "from unifiedmlm.models import list_models; from unifiedmlm.benchmarks import list_benchmarks; print(list_models(), list_benchmarks())"
# 期望输出：
# vllm x.y.z
# ['llava-1.5-7b'] ['mmbench']
```

> CUDA 版本不匹配 → 按 https://docs.vllm.ai/en/latest/getting_started/installation.html 选对应 wheel 重装。

## 4. HuggingFace 配置（国内必做）

国内访问 HF 慢，建议设镜像 + 把缓存挪到大盘。把下面这段写进 `~/.bashrc` 或者每次跑前 source 一下：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf_cache              # 改成你机器上的大盘路径
export HF_HUB_ENABLE_HF_TRANSFER=1         # 大文件加速
```

加速依赖：
```bash
pip install hf-transfer
```

如需私有 / gated 数据集：
```bash
huggingface-cli login
```

## 5. 预下载（强烈建议，避免首次跑时卡住）

```bash
# 1) LLaVA-1.5-7B 权重 (~14GB)
huggingface-cli download llava-hf/llava-1.5-7b-hf

# 2) MMBench 100 题子集（我们这版用的就是它，不是官方 lmms-lab/MMBench）
huggingface-cli download shijianS01/mmbench-subset --repo-type dataset

# 3) sentence-transformer（两步抽取 fallback 用）
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
```

校验数据集能加载：
```bash
python -c "from datasets import load_dataset; ds = load_dataset('shijianS01/mmbench-subset', split='eval'); print(len(ds), ds.column_names)"
# 期望：100 ['index', 'question', 'hint', 'answer', 'A', 'B', 'C', 'D', 'category', 'L2-category', 'image', 'source', 'comment', 'split']
```

## 6. 跑评测

直接用为这份子集准备的 config：

```bash
unifiedmlm-eval --config configs/eval/llava15_mmbench_subset.yaml
```

预期日志末尾：
```
[llava-1.5-7b @ mmbench] accuracy = 0.xxxx  (xx/100, extracted=xx)  elapsed=xx.xs
```

产出：
- `outputs/llava15_mmbench_subset/summary.json` — accuracy / 总数 / 抽取成功数 / 配置
- `outputs/llava15_mmbench_subset/per_sample.jsonl` — 每题 `gold / raw / predicted / extraction_method / extraction_score / correct`

## 7. 调参

编辑 `configs/eval/llava15_mmbench_subset.yaml`：

| 字段 | 何时改 |
|---|---|
| `model.config.tensor_parallel` | 多卡推理时调到卡数 |
| `model.config.gpu_memory_util` | OOM 时降到 0.85 |
| `model.config.max_model_len` | OOM 时降到 2048 |
| `model.config.sampling.temperature` | 默认 0（贪心，复现性强）；想看采样多样性才调高 |
| `batch_size` | OOM 时降 |
| `extractor.sim_threshold` | 抽取失败率高时降，误抽多时升 |

## 8. 常见报错

| 现象 | 排查 |
|---|---|
| `No matching distribution for numpy>=1.26.0` | Python 还是 3.8。检查 `which python` 和 `python --version`，必须 3.11 |
| `python3 --version` 是 3.8.10 | 正常 — 用 `python`（不带 3），别用 `python3` |
| `CondaError` 或 `conda: command not found` | `conda init bash` 后**重开终端** |
| `CUDA out of memory` | 降 `gpu_memory_util` → 0.85；降 `max_model_len` → 2048；降 `batch_size` |
| 模型输出乱码 / 死循环 | 检查 `sampling.max_tokens`（MCQ 任务 64–128 足够）；确认 prompt 模板没被改 |
| MMBench image 读不到 | HF 镜像问题，`export HF_HUB_ENABLE_HF_TRANSFER=1` 后重试 |
| `Unknown model 'xxx'` / `Unknown benchmark 'xxx'` | 没注册。看 README 的 "Extending" 节 |
| `NVIDIA driver too old (found version 12040)` | driver 只支持到 CUDA 12.4，但 pip 装了为更新 CUDA 编译的 vLLM/torch。见下文 §10 |
| 抽取 `method=="none"` 占比 > 10% | 看 `per_sample.jsonl` 的 `raw`，多半是 prompt 模板或采样温度问题 |

## 10. 驱动 / CUDA 不匹配

报错 `NVIDIA driver too old (found version 12040)` 意思不是 driver 真的旧，而是当前 driver 支持上限 CUDA **12.4**，而 `pip install vllm` 默认装了为 CUDA 12.6/12.8 编译的 wheel。

先看 driver 上限：
```bash
nvidia-smi | head -5     # 右上角 "CUDA Version: 12.x"
```

按 driver 上限选对应的 vLLM/torch 组合（requirements.txt 已锁 CUDA 12.4 的版本）：

| driver 上限 | vLLM | torch | torch wheel index |
|---|---|---|---|
| CUDA 12.4 | `vllm==0.8.5` | `torch==2.6.0` | `https://download.pytorch.org/whl/cu124` |
| CUDA 12.6 | `vllm==0.9.x` | `torch==2.7.x` | `https://download.pytorch.org/whl/cu126` |
| CUDA 12.8+ | 最新 vLLM | 最新 torch | 默认 PyPI 即可 |

降版本修复（不用动 driver）：
```bash
pip uninstall -y vllm torch torchvision torchaudio xformers flashinfer-python
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.5
```

升 driver 修复（需要 root + 重启，能跑最新 vLLM）：
```bash
sudo apt install -y nvidia-driver-570    # 或 driver-560 / driver-555
sudo reboot
```

## 9. 一键复跑

```bash
# 每次开新终端
conda activate unifiedmlm
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

cd UnifiedMLM
unifiedmlm-eval --config configs/eval/llava15_mmbench_subset.yaml
```
