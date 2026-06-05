"""Ovis 2.5 (structural embedding alignment, native-resolution ViT) via vLLM.

HF: AIDC-AI/Ovis2.5-{2B,9B}
vLLM: Ovis2_5（remote code 架构），0.10.1 起支持。
"""
from __future__ import annotations

from .registry import register_model
from .vllm_chat import VLLMChatModel


class _Ovis25Base(VLLMChatModel):
    STRIP_THINK = True  # Ovis2.5 有可选 thinking budget；默认关，strip 兜底


@register_model("ovis2.5-9b")
class Ovis25_9B(_Ovis25Base):
    DEFAULT_MODEL_PATH = "AIDC-AI/Ovis2.5-9B"


@register_model("ovis2.5-2b")
class Ovis25_2B(_Ovis25Base):
    DEFAULT_MODEL_PATH = "AIDC-AI/Ovis2.5-2B"
