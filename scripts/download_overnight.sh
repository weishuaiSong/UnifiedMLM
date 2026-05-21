#!/usr/bin/env bash
# Overnight bulk download — fail-soft, retry-friendly.
#
# 用法：
#   conda activate unifiedmlm-current
#   bash scripts/download_overnight.sh [tier1_lite|tier1|tier2|all]   # 默认 all
#
# 特性：
#   - 每个模型独立尝试，单个失败不中断后续（不 set -e）
#   - 每个模型最多重试 3 次（hf download 自带断点续传，重试只是再调一次）
#   - 详细日志 + 结尾汇总成功/失败清单
#
# 配合 tmux 使用（脚本末尾有指南）。

# -------- env --------
# 强制设到新位置（防止陷在旧 shell 还保留迁移前的 HF_HOME 值）。
export HF_HOME=/mnt/sdc/zimuwang/huggingface
export HUGGINGFACE_HUB_CACHE=/mnt/sdc/zimuwang/huggingface/hub
export HF_DATASETS_CACHE=/mnt/sdc/zimuwang/huggingface/datasets
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER=1
unset TRANSFORMERS_CACHE  # 防止陷阱（详见 docs/ENVS.md）

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: 'hf' CLI not found. 'conda activate unifiedmlm-current' 先。" >&2
  exit 1
fi

# -------- model lists --------
TIER1_LITE=(
  # "mistral-experimental/pixtral-12b"
  "deepseek-ai/deepseek-vl2-small"
)
TIER1_GATED=(
  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
)
TIER2=(
  "THUDM/glm-4v-9b"
  "microsoft/Phi-4-multimodal-instruct"
  "CohereLabs/aya-vision-8b"
  "moonshotai/Kimi-VL-A3B-Instruct"
)

case "${1:-all}" in
  tier1_lite) MODELS=("${TIER1_LITE[@]}") ;;
  tier1)      MODELS=("${TIER1_LITE[@]}" "${TIER1_GATED[@]}") ;;
  tier2)      MODELS=("${TIER2[@]}") ;;
  all)        MODELS=("${TIER1_LITE[@]}" "${TIER1_GATED[@]}" "${TIER2[@]}") ;;
  *) echo "usage: $0 {tier1_lite|tier1|tier2|all}" >&2; exit 2 ;;
esac

MAX_RETRIES=3
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs/logs"
mkdir -p "$LOG_DIR"
SUMMARY=()

echo "==================================================================="
echo " Overnight download — $(date '+%F %T')"
echo " HF_HOME=$HF_HOME"
echo " HF_ENDPOINT=$HF_ENDPOINT"
echo " ${#MODELS[@]} repos:"
for m in "${MODELS[@]}"; do echo "   - $m"; done
echo " Disk: $(df -h "$HF_HOME" | awk 'NR==2{print $4}') free on $(df -h "$HF_HOME" | awk 'NR==2{print $6}')"
echo "==================================================================="
echo

for repo in "${MODELS[@]}"; do
  echo "==================================================================="
  echo "==> [$(date '+%H:%M:%S')] $repo"
  echo "==================================================================="
  status="FAIL"
  for try in $(seq 1 "$MAX_RETRIES"); do
    echo "--- attempt $try/$MAX_RETRIES ---"
    if hf download "$repo"; then
      status="OK"
      break
    fi
    echo "--- attempt $try failed; sleeping 10s ---"
    sleep 10
  done

  size=$(du -sh "$HF_HOME/hub/models--$(echo "$repo" | tr '/' '-' | sed 's/^/-/' | sed 's/^-/models--/' )" 2>/dev/null | cut -f1)
  # 上面 sed 转换有点拐弯，下面更直接地查：
  size=$(du -sh "$HF_HOME/hub/models--${repo//\//--}" 2>/dev/null | cut -f1)
  SUMMARY+=("$status  $repo  (${size:-?})")
  echo "==> [$(date '+%H:%M:%S')] $repo -> $status (${size:-?})"
  echo "==> Disk free now: $(df -h "$HF_HOME" | awk 'NR==2{print $4}')"
  echo
done

echo "==================================================================="
echo " SUMMARY  $(date '+%F %T')"
echo "==================================================================="
printf '%s\n' "${SUMMARY[@]}"
echo "==================================================================="
ok_count=$(printf '%s\n' "${SUMMARY[@]}" | grep -c '^OK')
fail_count=$(printf '%s\n' "${SUMMARY[@]}" | grep -c '^FAIL')
echo " ${ok_count} ok / ${fail_count} fail / ${#MODELS[@]} total"
echo " Disk on $(df -h "$HF_HOME" | awk 'NR==2{print $6}'): $(df -h "$HF_HOME" | awk 'NR==2{print $4}') free"
