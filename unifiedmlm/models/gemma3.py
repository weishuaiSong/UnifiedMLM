"""Gemma 3 (multimodal, SigLIP + pan-and-scan) served by vLLM.

HF: google/gemma-3-{4b,12b}-it ⚠️ gated repo——下载需要 HF_TOKEN 且账号接受过
Google license（hf-mirror 也绕不开 gating）。
vLLM: Gemma3ForConditionalGeneration，0.8.0 起原生支持。
"""
from __future__ import annotations

from .registry import register_model
from .vllm_chat import VLLMChatModel


class _Gemma3Base(VLLMChatModel):
    # bf16 必须：Gemma 3 fp16 会溢出（官方卡片明确写了）
    EXTRA_LLM_KWARGS = {"dtype": "bfloat16"}


@register_model("gemma-3-4b")
class Gemma3_4B(_Gemma3Base):
    DEFAULT_MODEL_PATH = "google/gemma-3-4b-it"


@register_model("gemma-3-12b")
class Gemma3_12B(_Gemma3Base):
    DEFAULT_MODEL_PATH = "google/gemma-3-12b-it"
