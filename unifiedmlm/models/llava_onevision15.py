"""LLaVA-OneVision-1.5-8B-Instruct served by vLLM.

HF: lmms-lab/LLaVA-OneVision-1.5-8B-Instruct (released 2025-09)
Backbone: Qwen3-7B + SigLIP 2 (native-resolution).

Chat template (Qwen-style with LLaVA <image> placeholder):
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <image>
    {question}<|im_end|>
    <|im_start|>assistant
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import BaseVLMModel, VLMRequest, VLMResponse
from .registry import register_model


LLAVA_OV15_PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<image>\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@register_model("llava-onevision-1.5-8b")
class LLaVAOneVision15(BaseVLMModel):
    """LLaVA-OneVision-1.5-8B-Instruct wrapper over vLLM.

    Config keys:
      model_path:        HF repo or local path (default: lmms-lab/LLaVA-OneVision-1.5-8B-Instruct)
      tensor_parallel:   int, default 1
      max_model_len:     int, default 8192 (SigLIP 2 native-resolution → more visual tokens)
      gpu_memory_util:   float, default 0.9
      dtype:             str, default "auto"
      sampling:          dict passed to vllm.SamplingParams
    """

    DEFAULT_MODEL_PATH = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        from vllm import LLM, SamplingParams

        model_path = self.config.get("model_path", self.DEFAULT_MODEL_PATH)
        llm_kwargs: dict[str, Any] = dict(
            model=model_path,
            tensor_parallel_size=int(self.config.get("tensor_parallel", 1)),
            max_model_len=int(self.config.get("max_model_len", 8192)),
            gpu_memory_utilization=float(self.config.get("gpu_memory_util", 0.9)),
            dtype=self.config.get("dtype", "auto"),
            trust_remote_code=bool(self.config.get("trust_remote_code", True)),
            limit_mm_per_prompt={"image": 1},
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
                    f"LLaVA-OneVision-1.5 expects exactly 1 image per request, got {len(r.images)}"
                )
            prompt = LLAVA_OV15_PROMPT.format(question=r.prompt)
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
