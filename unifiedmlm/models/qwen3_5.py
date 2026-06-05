"""Qwen3.5 (natively multimodal) served by vLLM.

HF: Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B (released 2026; hybrid Gated-DeltaNet +
sparse-MoE, image/video input built in — there is no separate "-VL" line).

⚠️ vLLM 版本要求：**vLLM ≥ 0.17**（主 venv 锁的 0.11.2 跑不了）。
在 A100 等测试机上请单开 venv，见 docs/ENVS.md §"Qwen3.5 单独 venv"。

Prompt 不再手写：Qwen3.5 默认开 thinking mode，<think> 块的开关由 chat
template 的 `enable_thinking` kwarg 控制，手写模板容易跟官方 template 漂移。
这里用 AutoProcessor.apply_chat_template 渲染（含 vision 占位符），MCQ 评测
默认 `enable_thinking: false`。
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import BaseVLMModel, VLMRequest, VLMResponse
from .registry import register_model


def _strip_think(text: str) -> str:
    """Drop a leading <think>...</think> block if the model still emitted one."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


class _Qwen3_5Base(BaseVLMModel):
    """Qwen3.5 wrapper over vLLM (no HF fallback — vLLM supports it natively).

    Config keys:
      model_path:        HF repo or local path (default: DEFAULT_MODEL_PATH)
      tensor_parallel:   int, default 1
      max_model_len:     int, default 16384 (native context is 262k — eval 用不着)
      gpu_memory_util:   float, default 0.9
      dtype:             str, default "auto"
      max_pixels:        int, optional. Caps visual tokens via the dynamic-
                         resolution preprocessor, same knob as Qwen3-VL.
      enable_thinking:   bool, default False. Passed to the chat template;
                         True 时记得把 sampling.max_tokens 提到 ≥1024，
                         否则答案会被 <think> 吃掉。
      sampling:          dict passed to vllm.SamplingParams.
    """

    DEFAULT_MODEL_PATH: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        model_path = self.config.get("model_path", self.DEFAULT_MODEL_PATH)
        self._enable_thinking = bool(self.config.get("enable_thinking", False))
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        mm_processor_kwargs: dict[str, Any] = {}
        max_pixels = self.config.get("max_pixels")
        if max_pixels is not None:
            mm_processor_kwargs["max_pixels"] = int(max_pixels)

        self._llm = LLM(
            model=model_path,
            tensor_parallel_size=int(self.config.get("tensor_parallel", 1)),
            max_model_len=int(self.config.get("max_model_len", 16384)),
            gpu_memory_utilization=float(self.config.get("gpu_memory_util", 0.9)),
            dtype=self.config.get("dtype", "auto"),
            trust_remote_code=bool(self.config.get("trust_remote_code", True)),
            # 评测只喂单图；video=0 防止 video tower 抢 KV cache 预算
            # (同 Qwen2.5-VL 的坑，见 ENVS.md §4)。
            limit_mm_per_prompt={"image": 1, "video": 0},
            mm_processor_kwargs=mm_processor_kwargs or None,
        )
        sampling_cfg = dict(self.config.get("sampling") or {})
        sampling_cfg.setdefault("temperature", 0.0)
        sampling_cfg.setdefault("max_tokens", 1024 if self._enable_thinking else 128)
        self._sampling = SamplingParams(**sampling_cfg)

    def _render_prompt(self, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        try:
            return self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._enable_thinking,
            )
        except TypeError:
            # 老 template 不认 enable_thinking kwarg 时退化为默认行为，
            # 靠 _strip_think 兜底。
            return self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def generate(self, requests: Iterable[VLMRequest]) -> list[VLMResponse]:
        reqs = list(requests)
        if not reqs:
            return []

        vllm_inputs = []
        for r in reqs:
            if len(r.images) != 1:
                raise ValueError(
                    f"Qwen3.5 eval wrapper expects exactly 1 image per request, "
                    f"got {len(r.images)}"
                )
            vllm_inputs.append(
                {
                    "prompt": self._render_prompt(r.prompt),
                    "multi_modal_data": {"image": r.images[0]},
                }
            )

        outputs = self._llm.generate(vllm_inputs, sampling_params=self._sampling)
        responses: list[VLMResponse] = []
        for out in outputs:
            raw = out.outputs[0].text if out.outputs else ""
            responses.append(
                VLMResponse(
                    text=_strip_think(raw),
                    metadata={
                        "finish_reason": out.outputs[0].finish_reason if out.outputs else None,
                        "had_think_block": "</think>" in raw,
                    },
                )
            )
        return responses

    def shutdown(self) -> None:
        self._llm = None


@register_model("qwen3.5-4b")
class Qwen3_5_4B(_Qwen3_5Base):
    """Qwen3.5-4B — A100 可行性测试首选（单卡轻松）。"""

    DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-4B"


@register_model("qwen3.5-9b")
class Qwen3_5_9B(_Qwen3_5Base):
    """Qwen3.5-9B — 与现有 8B 级 VLM 同档可比。"""

    DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-9B"
