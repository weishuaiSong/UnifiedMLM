#!/usr/bin/env bash
# Download new VLM weights into HF_HOME for UnifiedMLM eval.
#
# Usage:
#   bash scripts/download_new_models.sh tier1_lite   # Pixtral 12B + DeepSeek-VL2-Small (~58 GB, 都不 gated)
#   bash scripts/download_new_models.sh tier1        # + Gemma 3 4B + 12B               (+~32 GB, gated)
#   bash scripts/download_new_models.sh tier2        # GLM-4V + Phi-4-MM + Aya + Kimi   (~78 GB, 都不 gated)
#   bash scripts/download_new_models.sh all          # tier1 + tier2                     (~168 GB)
#   bash scripts/download_new_models.sh <repo_id>    # 单个 repo
#
# Gated 状态（2026-05 实测）：
#   - mistral-experimental/pixtral-12b   → False ✓  (Pixtral 免 gated reupload)
#   - deepseek-ai/deepseek-vl2-small     → False ✓
#   - moonshotai/Kimi-VL-A3B-Instruct    → False ✓
#   - THUDM/glm-4v-9b                    → False ✓
#   - microsoft/Phi-4-multimodal-instruct → False ✓
#   - CohereLabs/aya-vision-8b           → **auto**   仅需 'hf auth login'（无需 accept license）
#   - google/gemma-3-*-it                → **manual** 需先到网页 https://huggingface.co/google/gemma-3-4b-it 点 Accept License
#
# 设 token: hf auth login   或   export HF_TOKEN=hf_xxx
#
# 全部走 HF_HOME 共享缓存（vLLM / transformers / 任意 env 都能读）。
# 默认走 hf-mirror.com；如要直连 huggingface.co，设 HF_USE_MIRROR=0。

set -euo pipefail

# ---------- env ----------
: "${HF_USE_MIRROR:=1}"
if [[ "$HF_USE_MIRROR" == "1" ]]; then
  : "${HF_ENDPOINT:=https://hf-mirror.com}"
else
  : "${HF_ENDPOINT:=https://huggingface.co}"
fi
: "${HF_HOME:=/mnt/sda/weishuaisong/huggingface}"
: "${HF_HUB_ENABLE_HF_TRANSFER:=1}"
export HF_ENDPOINT HF_HOME HF_HUB_ENABLE_HF_TRANSFER

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: 'hf' CLI not found. Activate env first:" >&2
  echo "  conda activate unifiedmlm-current  # or unifiedmlm" >&2
  exit 1
fi

# ---------- model lists ----------
# Tier 1 lite: 不需要 gated token，先下这两个能跑起来 wrapper。
TIER1_LITE=(
  "mistral-community/pixtral-12b"   # ~25 GB | Mistral Nemo backbone, mistralai 的免 gated reupload
  "deepseek-ai/deepseek-vl2-small"  # ~33 GB | DeepSeek-MoE 16B/2.4B-active, MMBench 83.1
)
# Gemma 3 必须 gated → 需要 HF_TOKEN
TIER1_GATED=(
  "google/gemma-3-4b-it"            # ~8 GB  | Gemma 3 backbone, native MM, 128K ctx
  "google/gemma-3-12b-it"           # ~24 GB | Gemma 3 主力档
)
TIER2=(
  "THUDM/glm-4v-9b"                       # ~18 GB | ChatGLM4 backbone, 中文
  "microsoft/Phi-4-multimodal-instruct"   # ~12 GB | Phi-4 + audio modality
  "CohereLabs/aya-vision-8b"              # ~16 GB | Cohere Command, 23 lang
  "moonshotai/Kimi-VL-A3B-Instruct"       # ~32 GB | MoE 16B/3B-active
)

# ---------- pick set ----------
case "${1:-}" in
  tier1_lite) MODELS=("${TIER1_LITE[@]}") ;;
  tier1)      MODELS=("${TIER1_LITE[@]}" "${TIER1_GATED[@]}") ;;
  tier2)      MODELS=("${TIER2[@]}") ;;
  all)        MODELS=("${TIER1_LITE[@]}" "${TIER1_GATED[@]}" "${TIER2[@]}") ;;
  "")         echo "usage: $0 {tier1_lite|tier1|tier2|all|<repo_id>}" >&2; exit 2 ;;
  */*)        MODELS=("$1") ;;
  *)          echo "unknown preset: $1" >&2; exit 2 ;;
esac

# ---------- token sanity for gated ----------
need_token=0
for m in "${MODELS[@]}"; do
  case "$m" in
    google/gemma-3-*|mistralai/Pixtral-*|meta-llama/*) need_token=1 ;;
  esac
done
if [[ "$need_token" == "1" && -z "${HF_TOKEN:-}" && ! -f "$HOME/.cache/huggingface/token" ]]; then
  echo "WARNING: 这批包含 gated repo，但没检测到 HF_TOKEN / hf auth。" >&2
  echo "         先 'hf auth login' 或 'export HF_TOKEN=hf_xxx'，并到模型主页点 Accept License。" >&2
  echo "         （Gemma/Llama 系全部 gated；Pixtral 用了 mistral-community 镜像可免）" >&2
  echo
fi

# ---------- disk preflight ----------
echo "==> HF_HOME=$HF_HOME"
echo "==> HF_ENDPOINT=$HF_ENDPOINT"
echo "==> Disk on $(df -h "$HF_HOME" | awk 'NR==2{print $6}'): $(df -h "$HF_HOME" | awk 'NR==2{print $4}') free"
echo "==> About to download ${#MODELS[@]} repo(s):"
for m in "${MODELS[@]}"; do echo "    - $m"; done
echo

# ---------- download loop ----------
for repo in "${MODELS[@]}"; do
  echo "==================================================================="
  echo "==> [$(date +%H:%M:%S)] hf download $repo"
  echo "==================================================================="
  # 不指定 --local-dir → 走 HF_HOME/hub 共享缓存。
  hf download "$repo"
  echo "==> done: $repo"
  echo "==> disk free now: $(df -h "$HF_HOME" | awk 'NR==2{print $4}')"
  echo
done

echo "==> All downloads complete."
