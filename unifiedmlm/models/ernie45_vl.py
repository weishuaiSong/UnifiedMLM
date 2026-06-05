"""ERNIE 4.5 VL 28B-A3B (Baidu, MoE 激活 3B) via vLLM.

HF: baidu/ERNIE-4.5-VL-28B-A3B-PT（PT = post-trained 非 thinking 版）
vLLM: Ernie4_5_VLMoeForConditionalGeneration，0.10.x 起支持。
权重 ~56G bf16，A800 80G 单卡可放；KV 余量小，gpu_memory_util 默认调高。
"""
from __future__ import annotations

from .registry import register_model
from .vllm_chat import VLLMChatModel


@register_model("ernie-4.5-vl-28b-a3b")
class Ernie45VL(VLLMChatModel):
    DEFAULT_MODEL_PATH = "baidu/ERNIE-4.5-VL-28B-A3B-PT"
    DEFAULT_LIMIT_MM = {"image": 1, "video": 0}
    EXTRA_LLM_KWARGS = {"gpu_memory_utilization": 0.95}
