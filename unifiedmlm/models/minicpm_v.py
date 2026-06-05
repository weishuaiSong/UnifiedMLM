"""MiniCPM-V 4.5 (8B, Qwen3 + SigLIP2) served by vLLM.

HF: openbmb/MiniCPM-V-4_5
vLLM: MiniCPMV（remote code 架构），0.10.1 起支持 4.5。
混合 thinking：⚠️ 实测默认**开**（与卡片宣传相反）——MCQ 评测必须
chat_template_kwargs: {enable_thinking: false}，否则 <think> 吃光 token 预算；
STRIP_THINK + 大 max_tokens 兜底。
另：trf 5.x 下 remote tokenizer 类不加载（im_start_id 缺失），需要 trf 4.x。
"""
from __future__ import annotations

from .registry import register_model
from .vllm_chat import VLLMChatModel


@register_model("minicpm-v-4.5")
class MiniCPMV45(VLLMChatModel):
    DEFAULT_MODEL_PATH = "openbmb/MiniCPM-V-4_5"
    DEFAULT_LIMIT_MM = {"image": 1, "video": 0}
    STRIP_THINK = True  # 兜底；默认不思考时无害
