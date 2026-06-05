"""Eagle 2.5-8B (NVIDIA, long-context VLM) via vLLM.

HF: nvidia/Eagle2.5-8B
vLLM: Eagle2_5_VLForConditionalGeneration（remote code 架构）。
"""
from __future__ import annotations

from .registry import register_model
from .vllm_chat import VLLMChatModel


@register_model("eagle2.5-8b")
class Eagle25(VLLMChatModel):
    DEFAULT_MODEL_PATH = "nvidia/Eagle2.5-8B"
    DEFAULT_LIMIT_MM = {"image": 1, "video": 0}
