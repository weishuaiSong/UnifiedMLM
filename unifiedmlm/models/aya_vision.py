"""Aya Vision 8B (Cohere, C4AI Command-R7B + SigLIP2) via vLLM.

HF: CohereLabs/aya-vision-8b ⚠️ gated（CC-BY-NC + 接受条款），下载需 HF_TOKEN。
vLLM: AyaVisionForConditionalGeneration，0.8.x 起原生支持。
"""
from __future__ import annotations

from .registry import register_model
from .vllm_chat import VLLMChatModel


@register_model("aya-vision-8b")
class AyaVision8B(VLLMChatModel):
    DEFAULT_MODEL_PATH = "CohereLabs/aya-vision-8b"
