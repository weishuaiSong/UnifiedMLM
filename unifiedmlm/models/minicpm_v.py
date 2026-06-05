"""MiniCPM-V 4.5 (8B, Qwen3 + SigLIP2) served by vLLM.

HF: openbmb/MiniCPM-V-4_5
vLLM: MiniCPMV（remote code 架构），0.10.1 起支持 4.5。
混合 thinking：默认关；要开在 yaml 给 chat_template_kwargs: {enable_thinking: true}
并把 sampling.max_tokens 提到 ≥2048。
"""
from __future__ import annotations

from .registry import register_model
from .vllm_chat import VLLMChatModel


@register_model("minicpm-v-4.5")
class MiniCPMV45(VLLMChatModel):
    DEFAULT_MODEL_PATH = "openbmb/MiniCPM-V-4_5"
    DEFAULT_LIMIT_MM = {"image": 1, "video": 0}
    STRIP_THINK = True  # 兜底；默认不思考时无害
