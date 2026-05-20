"""Qwen2.5-VL-7B-Instruct served by vLLM.

Reference prompt format (Qwen2.5-VL chat template):
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>
    <|im_start|>assistant
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import BaseVLMModel, VLMRequest, VLMResponse
from .registry import register_model


QWEN25_VL_PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@register_model("qwen2.5-vl-7b")
class Qwen25VL(BaseVLMModel):
    """Qwen2.5-VL-7B-Instruct wrapper over vLLM.

    Config keys:
      model_path:        HF repo or local path (default: Qwen/Qwen2.5-VL-7B-Instruct)
      tensor_parallel:   int, default 1
      max_model_len:     int, default 8192 (Qwen2.5-VL handles higher resolutions
                         that produce more visual tokens than LLaVA's 576)
      gpu_memory_util:   float, default 0.9
      dtype:             str, default "auto"
      max_pixels:        int, optional. Caps visual tokens by limiting the image
                         resolution Qwen2.5-VL's dynamic-resolution preprocessor
                         will accept. None = use processor default (~12845056).
                         Set lower (e.g. 1003520 ≈ 1280×784) to reduce KV cache
                         pressure during high-batch eval.
      sampling:          dict passed to vllm.SamplingParams.
    """

    DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        from vllm import LLM, SamplingParams

        model_path = self.config.get("model_path", self.DEFAULT_MODEL_PATH)
        mm_processor_kwargs: dict[str, Any] = {}
        max_pixels = self.config.get("max_pixels")
        if max_pixels is not None:
            mm_processor_kwargs["max_pixels"] = int(max_pixels)

        llm_kwargs: dict[str, Any] = dict(
            model=model_path,
            tensor_parallel_size=int(self.config.get("tensor_parallel", 1)),
            max_model_len=int(self.config.get("max_model_len", 8192)),
            gpu_memory_utilization=float(self.config.get("gpu_memory_util", 0.9)),
            dtype=self.config.get("dtype", "auto"),
            trust_remote_code=bool(self.config.get("trust_remote_code", True)),
            limit_mm_per_prompt={"image": 1, "video": 0},
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
                    f"Qwen2.5-VL expects exactly 1 image per request, got {len(r.images)}"
                )
            prompt = QWEN25_VL_PROMPT.format(question=r.prompt)
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
