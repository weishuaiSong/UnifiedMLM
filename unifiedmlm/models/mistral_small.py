"""Mistral Small 3.1 24B Instruct (vision) via vLLM.

HF: mistralai/Mistral-Small-3.1-24B-Instruct-2503
vLLM: Mistral3ForConditionalGeneration（HF 格式），0.8.x 起支持。
官方推荐 mistral 格式三件套（tokenizer_mode/config_format/load_format =
"mistral"）；先走 HF 格式，报 tokenizer/template 错时在 yaml 里加：
  llm_kwargs: {tokenizer_mode: mistral, config_format: mistral, load_format: mistral}

24B bf16 ≈ 48G —— 4090 跑不了，A800/A100 80G 单卡可以。
"""
from __future__ import annotations

from .registry import register_model
from .vllm_chat import VLLMChatModel


@register_model("mistral-small-3.1-24b")
class MistralSmall31(VLLMChatModel):
    DEFAULT_MODEL_PATH = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
