"""LLaVA-1.5 (HF weights) served by vLLM.

Reference prompt format for llava-hf/llava-1.5-7b-hf:
    "USER: <image>\n{question}\nASSISTANT:"
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import BaseVLMModel, VLMRequest, VLMResponse
from .registry import register_model


LLAVA_15_PROMPT = "USER: <image>\n{question}\nASSISTANT:"


@register_model("llava-1.5-7b")
class LLaVA15(BaseVLMModel):
    """LLaVA-1.5-7B wrapper over vLLM.

    Config keys:
      model_path:        HF repo or local path (default: llava-hf/llava-1.5-7b-hf)
      tensor_parallel:   int, default 1
      max_model_len:     int, default 4096
      gpu_memory_util:   float, default 0.9
      dtype:             str, default "auto"
      sampling:          dict passed to vllm.SamplingParams (temperature, top_p, max_tokens, ...)
    """

    DEFAULT_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        # Imported lazily so the package can be inspected without vLLM installed.
        from vllm import LLM, SamplingParams

        model_path = self.config.get("model_path", self.DEFAULT_MODEL_PATH)
        self._llm = LLM(
            model=model_path,
            tensor_parallel_size=int(self.config.get("tensor_parallel", 1)),
            max_model_len=int(self.config.get("max_model_len", 4096)),
            gpu_memory_utilization=float(self.config.get("gpu_memory_util", 0.9)),
            dtype=self.config.get("dtype", "auto"),
            trust_remote_code=bool(self.config.get("trust_remote_code", False)),
            limit_mm_per_prompt={"image": 1},
        )
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
                    f"LLaVA-1.5 expects exactly 1 image per request, got {len(r.images)}"
                )
            prompt = LLAVA_15_PROMPT.format(question=r.prompt)
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
            responses.append(VLMResponse(text=text, metadata={"finish_reason": out.outputs[0].finish_reason if out.outputs else None}))
        return responses

    def shutdown(self) -> None:
        # vLLM cleans up on GC; nothing extra here for now.
        self._llm = None
