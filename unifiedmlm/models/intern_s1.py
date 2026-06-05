"""Intern-S1-mini (8B, Qwen3-8B + InternViT，科学多模态) via vLLM.

HF: internlm/Intern-S1-mini
vLLM: InternS1ForConditionalGeneration，0.10.x 起支持。
默认开 thinking——yaml 里 chat_template_kwargs: {enable_thinking: false} 关掉
（MCQ 评测默认），STRIP_THINK 兜底。
"""
from __future__ import annotations

from .registry import register_model
from .vllm_chat import VLLMChatModel


@register_model("intern-s1-mini")
class InternS1Mini(VLLMChatModel):
    DEFAULT_MODEL_PATH = "internlm/Intern-S1-mini"
    DEFAULT_LIMIT_MM = {"image": 1, "video": 0}
    DEFAULT_MAX_TOKENS = 2048
    STRIP_THINK = True
