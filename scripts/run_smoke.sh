#!/usr/bin/env bash
# Quick smoke test: LLaVA-1.5-7B on 50 MMBench samples.
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

OUT_DIR="outputs/smoke_$(date +%Y%m%d_%H%M%S)"

unifiedmlm-eval \
  --model llava-1.5-7b \
  --benchmark mmbench \
  --limit 50 \
  --batch-size 8 \
  --output-dir "${OUT_DIR}"

echo "Done. Outputs in ${OUT_DIR}"
