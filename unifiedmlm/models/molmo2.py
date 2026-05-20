"""Molmo 2 (AI2, released 2025-12-16) served by vLLM.

HF (确认过的实际命名):
  - allenai/Molmo2-8B       (Qwen3-8B          + SigLIP 2)
  - allenai/Molmo2-O-7B     (Olmo3-7B-Instruct + SigLIP 2)   -- 唯一非 Qwen-family LM
  - allenai/Molmo2-4B       (Qwen3-4B-Instruct + SigLIP 2)   -- pilot 用

Molmo 2 supports multi-image + video + grounding; we only use single-image for
template scaling experiments.

Chat template:
  - 8B / 4B (Qwen3 backbone): Qwen3 chat format.
  - O-7B (Olmo3 backbone): OLMo chat format (uses `<|user|>` / `<|assistant|>` markers).

注意：Molmo 2 的 vLLM 集成在 2026-05 时尚在迭代；如果 vLLM 不支持，
建议 fallback 到 transformers backend（见 docs/SETUP.md）。
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import BaseVLMModel, VLMRequest, VLMResponse
from .registry import register_model


# Qwen3-style template used by Molmo2-8B and Molmo2-4B.
MOLMO2_QWEN3_PROMPT = (
    "<|im_start|>system\nYou are Molmo, a helpful vision-language assistant.<|im_end|>\n"
    "<|im_start|>user\n<image>\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# OLMo chat template used by Molmo2-O-7B.
MOLMO2_OLMO_PROMPT = (
    "<|system|>\nYou are Molmo, a helpful vision-language assistant.\n"
    "<|user|>\n<image>\n{question}\n"
    "<|assistant|>\n"
)


class _Molmo2Base(BaseVLMModel):
    """Shared vLLM init for Molmo 2 variants.

    Subclasses set:
      DEFAULT_MODEL_PATH (HF id)
      PROMPT_TEMPLATE    (chat template with {question} placeholder + <image>)

    Molmo 2 在当前 vLLM release 上需要 hf_overrides 显式告诉它 architecture，
    否则报 "Unknown architecture"。默认值已经包好；YAML 里 hf_overrides 可覆盖。
    """

    DEFAULT_MODEL_PATH: str = ""
    PROMPT_TEMPLATE: str = ""

    # Default hf_overrides for Molmo 2 in current vLLM releases.
    # 如果未来 vLLM 原生识别 Molmo2 arch，可在 YAML 里设 hf_overrides: null 关掉。
    DEFAULT_HF_OVERRIDES: dict[str, Any] = {
        "architectures": ["Molmo2ForConditionalGeneration"],
        "is_mm_prefix_lm": False,
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        from vllm import LLM, SamplingParams

        model_path = self.config.get("model_path", self.DEFAULT_MODEL_PATH)

        # hf_overrides: config 里显式给值则用 config；未给则用 DEFAULT；
        # 显式给 None / null 表示不传 override（用于 vLLM 已原生识别的情况）
        if "hf_overrides" in self.config:
            hf_overrides = self.config["hf_overrides"]
        else:
            hf_overrides = dict(self.DEFAULT_HF_OVERRIDES)

        llm_kwargs: dict[str, Any] = dict(
            model=model_path,
            tensor_parallel_size=int(self.config.get("tensor_parallel", 1)),
            max_model_len=int(self.config.get("max_model_len", 8192)),
            gpu_memory_utilization=float(self.config.get("gpu_memory_util", 0.9)),
            dtype=self.config.get("dtype", "auto"),
            trust_remote_code=bool(self.config.get("trust_remote_code", True)),
            limit_mm_per_prompt={"image": 1},
        )
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
                    f"{self.name} expects exactly 1 image per request, got {len(r.images)}"
                )
            prompt = self.PROMPT_TEMPLATE.format(question=r.prompt)
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


@register_model("molmo2-8b")
class Molmo2_8B(_Molmo2Base):
    """Molmo2-8B (Qwen3-8B backbone + SigLIP 2)."""

    DEFAULT_MODEL_PATH = "allenai/Molmo2-8B"
    PROMPT_TEMPLATE = MOLMO2_QWEN3_PROMPT


@register_model("molmo2-o-7b")
class Molmo2_O_7B(_Molmo2Base):
    """Molmo2-O-7B with Olmo3-7B-Instruct backbone.

    关键 cross-arch 模型：当前阵容里唯一非 Qwen-family LM + 完全开放训练数据。
    "O" 取自 OLMo（Ai2 自家完全开放的 LM）。
    """

    DEFAULT_MODEL_PATH = "allenai/Molmo2-O-7B"
    PROMPT_TEMPLATE = MOLMO2_OLMO_PROMPT


@register_model("molmo2-4b")
class Molmo2_4B(_Molmo2Base):
    """Molmo2-4B (Qwen3-4B backbone). 适合 pilot / scaling-with-model-size 对照。"""

    DEFAULT_MODEL_PATH = "allenai/Molmo2-4B"
    PROMPT_TEMPLATE = MOLMO2_QWEN3_PROMPT
