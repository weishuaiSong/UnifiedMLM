"""InternVL3.5-8B served by vLLM.

HF: OpenGVLab/InternVL3_5-8B (released 2025-08; supersedes InternVL3-9B for our use)
Backbone: Qwen3 LM + InternViT-300M (dynamic patching + optional ViR efficient routing).

Chat template (InternVL-style):
    <|im_start|>system
    你是由上海人工智能实验室联合商汤科技研发的书生多模态大模型，英文名叫 InternVL,
    是一个有用无害的人工智能助手。<|im_end|>
    <|im_start|>user
    <image>
    {question}<|im_end|>
    <|im_start|>assistant
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import BaseVLMModel, VLMRequest, VLMResponse
from .registry import register_model


# Use English system prompt for consistency with other wrappers.
INTERNVL3_5_PROMPT = (
    "<|im_start|>system\nYou are InternVL, a helpful vision-language assistant developed by Shanghai AI Lab.<|im_end|>\n"
    "<|im_start|>user\n<image>\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@register_model("internvl3.5-8b")
class InternVL35(BaseVLMModel):
    """InternVL3.5-8B wrapper over vLLM.

    Config keys:
      model_path:        HF repo or local path (default: OpenGVLab/InternVL3_5-8B)
      tensor_parallel:   int, default 1
      max_model_len:     int, default 16384 (InternViT dynamic patching can yield many tokens)
      gpu_memory_util:   float, default 0.9
      dtype:             str, default "auto"
      max_dynamic_patch: int, optional. Caps visual tokens via InternVL's dynamic
                         tiling (default 12 in HF processor). Lower → fewer tokens,
                         less KV pressure.
      sampling:          dict passed to vllm.SamplingParams
    """

    DEFAULT_MODEL_PATH = "OpenGVLab/InternVL3_5-8B"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        from vllm import LLM, SamplingParams

        model_path = self.config.get("model_path", self.DEFAULT_MODEL_PATH)
        mm_processor_kwargs: dict[str, Any] = {}
        max_dynamic_patch = self.config.get("max_dynamic_patch")
        if max_dynamic_patch is not None:
            mm_processor_kwargs["max_dynamic_patch"] = int(max_dynamic_patch)

        llm_kwargs: dict[str, Any] = dict(
            model=model_path,
            tensor_parallel_size=int(self.config.get("tensor_parallel", 1)),
            max_model_len=int(self.config.get("max_model_len", 16384)),
            gpu_memory_utilization=float(self.config.get("gpu_memory_util", 0.9)),
            dtype=self.config.get("dtype", "auto"),
            trust_remote_code=bool(self.config.get("trust_remote_code", True)),
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs=mm_processor_kwargs or None,
        )
        hf_overrides = self.config.get("hf_overrides")
        if hf_overrides:
            llm_kwargs["hf_overrides"] = hf_overrides
        self._llm = LLM(**llm_kwargs)
        sampling_cfg = dict(self.config.get("sampling") or {})
        sampling_cfg.setdefault("temperature", 0.0)
        sampling_cfg.setdefault("max_tokens", 256)
        self._sampling = SamplingParams(**sampling_cfg)

    def generate(self, requests: Iterable[VLMRequest]) -> list[VLMResponse]:
        reqs = list(requests)
        if not reqs:
            return []

        vllm_inputs = []
        for r in reqs:
            if len(r.images) != 1:
                raise ValueError(
                    f"InternVL3.5 expects exactly 1 image per request, got {len(r.images)}"
                )
            prompt = INTERNVL3_5_PROMPT.format(question=r.prompt)
            vllm_inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": r.images[0]},
                }
            )

        outputs = self._llm.generate(vllm_inputs, sampling_params=self._sampling)
        responses: list[VLMResponse] = []
        for out in outputs:
            text = out.outputs[0].text.strip() if out.outputs else ""
            responses.append(
                VLMResponse(
                    text=text,
                    metadata={"finish_reason": out.outputs[0].finish_reason if out.outputs else None},
                )
            )
        return responses

    def shutdown(self) -> None:
        self._llm = None
