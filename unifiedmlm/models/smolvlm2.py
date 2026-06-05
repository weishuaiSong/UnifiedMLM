"""SmolVLM2-2.2B (HuggingFace, SigLIP + SmolLM2) via vLLM.

HF: HuggingFaceTB/SmolVLM2-2.2B-Instruct
vLLM: SmolVLMForConditionalGeneration，0.7.x 起（Idefics3 系）。
模型谱系里的 small 端点，用于 scaling 维度。
"""
from __future__ import annotations

from .registry import register_model
from .vllm_chat import VLLMChatModel


@register_model("smolvlm2-2.2b")
class SmolVLM22(VLLMChatModel):
    DEFAULT_MODEL_PATH = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    DEFAULT_LIMIT_MM = {"image": 1, "video": 0}
