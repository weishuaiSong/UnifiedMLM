"""GLM-4.1V-9B-Thinking (GLM-4-9B-0414 + AIMv2-Huge) via vLLM.

HF: zai-org/GLM-4.1V-9B-Thinking
vLLM: Glm4vForConditionalGeneration，0.10.0 起原生支持。

总是输出 <think>...</think><answer>...</answer>，没有关闭开关——
STRIP_THINK 取 <answer> 内文本；max_tokens 默认给 2048 防止答案被
思考链吃掉。比库里已有的 GLM-4V-9B（ChatGLM 底座、HF backend 150s/100q）
新一代且快得多，跑通后可替代之。
"""
from __future__ import annotations

from .registry import register_model
from .vllm_chat import VLLMChatModel


@register_model("glm-4.1v-9b-thinking")
class GLM41VThinking(VLLMChatModel):
    DEFAULT_MODEL_PATH = "zai-org/GLM-4.1V-9B-Thinking"
    DEFAULT_LIMIT_MM = {"image": 1, "video": 0}
    DEFAULT_MAX_TOKENS = 2048
    STRIP_THINK = True
