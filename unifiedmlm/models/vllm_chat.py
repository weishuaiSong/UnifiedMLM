"""Shared base for vLLM-native models via the `LLM.chat()` API.

2026-06 批量接入 vLLM 原生 VLM 时引入。之前的 wrapper（qwen3_vl 等）手写
prompt 模板或用 AutoProcessor.apply_chat_template + llm.generate；但 11 个
异构家族的 vision placeholder 注入规则各不相同（<image>、<|image_pad|>、
[IMG]、<image>./</image>...），手写极易错。vLLM 的 chat 工具链对每个已支持
架构注册了正确的 placeholder 注入逻辑，`llm.chat()` 直接复用——每个模型
只剩一层注册壳（见 gemma3.py 等）。

图像以 base64 PNG data URL 传入（chat API 的标准多模态入口，全架构通用）；
单图 MCQ 评测下编码开销可忽略。

子类可覆盖的类属性：
  DEFAULT_MODEL_PATH    HF repo
  DEFAULT_MAX_MODEL_LEN 默认 8192（mmbench prompt 短）
  DEFAULT_MAX_TOKENS    默认 128；thinking 模型设 ≥2048
  DEFAULT_LIMIT_MM      默认 {"image": 1}；有 video tower 的设 {"image":1,"video":0}
  EXTRA_LLM_KWARGS      额外 LLM(...) kwargs（如 tokenizer_mode="mistral"）
  STRIP_THINK           True 时剥 <think>/<answer> 块再返回

Config keys（全部 yaml 可覆盖）：
  model_path / tensor_parallel / max_model_len / gpu_memory_util / dtype /
  trust_remote_code / limit_mm_per_prompt / mm_processor_kwargs /
  chat_template_kwargs（如 {"enable_thinking": false}）/
  llm_kwargs（兜底 passthrough，最后 merge 进 LLM(...)）/ sampling
"""
from __future__ import annotations

import base64
import io
import re
from typing import Any, Iterable

from .base import BaseVLMModel, VLMRequest, VLMResponse


def _pil_to_data_url(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _strip_think(text: str) -> str:
    """剥 thinking 模型的推理块：<think>...</think> 之后取正文；
    GLM-4.1V / Keye 风格的 <answer>...</answer> 取标签内文本。"""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S)
    if m:
        text = m.group(1)
    return text.strip()


class VLLMChatModel(BaseVLMModel):
    DEFAULT_MODEL_PATH: str = ""
    DEFAULT_MAX_MODEL_LEN: int = 8192
    DEFAULT_MAX_TOKENS: int = 128
    DEFAULT_LIMIT_MM: dict[str, int] = {"image": 1}
    EXTRA_LLM_KWARGS: dict[str, Any] = {}
    STRIP_THINK: bool = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        from vllm import LLM, SamplingParams

        model_path = self.config.get("model_path", self.DEFAULT_MODEL_PATH)
        llm_kwargs: dict[str, Any] = dict(
            model=model_path,
            tensor_parallel_size=int(self.config.get("tensor_parallel", 1)),
            max_model_len=int(self.config.get("max_model_len", self.DEFAULT_MAX_MODEL_LEN)),
            gpu_memory_utilization=float(self.config.get("gpu_memory_util", 0.9)),
            dtype=self.config.get("dtype", "auto"),
            trust_remote_code=bool(self.config.get("trust_remote_code", True)),
            limit_mm_per_prompt=dict(
                self.config.get("limit_mm_per_prompt", self.DEFAULT_LIMIT_MM)
            ),
        )
        mm_processor_kwargs = self.config.get("mm_processor_kwargs")
        if mm_processor_kwargs:
            llm_kwargs["mm_processor_kwargs"] = dict(mm_processor_kwargs)
        llm_kwargs.update(self.EXTRA_LLM_KWARGS)
        # 兜底 passthrough：yaml 里给 llm_kwargs 可覆盖以上任何一项
        llm_kwargs.update(dict(self.config.get("llm_kwargs") or {}))
        self._llm = LLM(**llm_kwargs)

        self._chat_template_kwargs = dict(self.config.get("chat_template_kwargs") or {})
        sampling_cfg = dict(self.config.get("sampling") or {})
        sampling_cfg.setdefault("temperature", 0.0)
        sampling_cfg.setdefault("max_tokens", self.DEFAULT_MAX_TOKENS)
        self._sampling = SamplingParams(**sampling_cfg)

    def generate(self, requests: Iterable[VLMRequest]) -> list[VLMResponse]:
        reqs = list(requests)
        if not reqs:
            return []

        conversations = []
        for r in reqs:
            if len(r.images) != 1:
                raise ValueError(
                    f"{self.name} expects exactly 1 image per request, got {len(r.images)}"
                )
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": _pil_to_data_url(r.images[0])},
                            },
                            {"type": "text", "text": r.prompt},
                        ],
                    }
                ]
            )

        chat_kwargs: dict[str, Any] = dict(sampling_params=self._sampling, use_tqdm=True)
        if self._chat_template_kwargs:
            chat_kwargs["chat_template_kwargs"] = self._chat_template_kwargs
        try:
            outputs = self._llm.chat(conversations, **chat_kwargs)
        except TypeError:
            # 老 vLLM 的 chat() 不认 chat_template_kwargs —— 去掉重试，
            # thinking 块靠 STRIP_THINK 兜底
            chat_kwargs.pop("chat_template_kwargs", None)
            outputs = self._llm.chat(conversations, **chat_kwargs)

        responses: list[VLMResponse] = []
        for out in outputs:
            raw = out.outputs[0].text if out.outputs else ""
            text = _strip_think(raw) if self.STRIP_THINK else raw.strip()
            responses.append(
                VLMResponse(
                    text=text,
                    metadata={
                        "finish_reason": out.outputs[0].finish_reason if out.outputs else None,
                        "had_think_block": "</think>" in raw,
                    },
                )
            )
        return responses

    def shutdown(self) -> None:
        self._llm = None
