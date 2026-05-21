"""Phi-4 Multimodal 5.6B (Microsoft, vision + audio) over vLLM.

HF: microsoft/Phi-4-multimodal-instruct
vLLM 0.11.2 原生支持 arch Phi4MMForCausalLM。

Chat prompt (Phi-4 convention, single-image variant):
    <|user|><|image_1|>{question}<|end|><|assistant|>
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import BaseVLMModel, VLMRequest, VLMResponse
from .registry import register_model


PHI4_MM_PROMPT = "<|user|><|image_1|>{question}<|end|><|assistant|>"


@register_model("phi-4-multimodal")
class Phi4Multimodal(BaseVLMModel):
    """Phi-4 Multimodal 5.6B wrapper over vLLM.

    Config keys:
      model_path:      HF repo / local path (default: microsoft/Phi-4-multimodal-instruct)
      tensor_parallel: int, default 1
      max_model_len:   int, default 8192
      gpu_memory_util: float, default 0.9
      dtype:           str, default "auto"
      sampling:        dict (temperature, max_tokens)
    """

    DEFAULT_MODEL_PATH = "microsoft/Phi-4-multimodal-instruct"

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
                    f"Phi-4-MM expects exactly 1 image per request, got {len(r.images)}"
                )
            vllm_inputs.append(
                {
                    "prompt": PHI4_MM_PROMPT.format(question=r.prompt),
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
