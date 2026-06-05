"""Keye-VL-1.5-8B (快手, Qwen3-8B + SigLIP, slow-fast video encoding) via vLLM.

HF: Kwai-Keye/Keye-VL-1.5-8B
vLLM: KeyeVL1_5ForConditionalGeneration，0.10.x 起支持。
auto-thinking 模型：会自行决定是否输出 <think>/<analysis> 块，
STRIP_THINK + 加大 max_tokens 兜底。
"""
from __future__ import annotations

from .registry import register_model
from .vllm_chat import VLLMChatModel


@register_model("keye-vl-1.5-8b")
class KeyeVL15(VLLMChatModel):
    DEFAULT_MODEL_PATH = "Kwai-Keye/Keye-VL-1.5-8B"
    DEFAULT_LIMIT_MM = {"image": 1, "video": 0}
    DEFAULT_MAX_TOKENS = 2048
    STRIP_THINK = True
