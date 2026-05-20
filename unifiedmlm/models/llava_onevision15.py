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
    """LLaVA-OneVision-1.5-8B-Instruct wrapper.

    Backend 选项（config["backend"]）：
      - "vllm" (default): vLLM 后端；需要 vLLM 已合并 LLaVAOneVision1_5 架构（截至
                        2026-05 vLLM 0.11.2 还没合并）。
      - "hf":             transformers AutoModelForCausalLM + trust_remote_code。
                        模型的 chat template 走 qwen_vl_utils.process_vision_info
                        （HF 卡描述里推荐的方式），不依赖 vLLM 支持。

    Config keys:
      model_path:        HF repo or local path (default: lmms-lab/LLaVA-OneVision-1.5-8B-Instruct)
      backend:           "vllm" | "hf"
      tensor_parallel:   int, default 1                (vllm only)
      max_model_len:     int, default 8192             (vllm only)
      gpu_memory_util:   float, default 0.9            (vllm only)
      dtype:             "bfloat16" | "float16" | "auto"
      device:            cuda device for hf backend (default cuda:0)
      sampling:          dict (temperature, max_tokens)
    """

    DEFAULT_MODEL_PATH = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._backend = str(self.config.get("backend", "vllm")).lower()
        if self._backend == "hf":
            self._init_hf()
        else:
            self._init_vllm()

    def _init_vllm(self) -> None:
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

    def _init_hf(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_path = self.config.get("model_path", self.DEFAULT_MODEL_PATH)
        dtype_name = str(self.config.get("dtype", "bfloat16")).lower()
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "auto": "auto"}
        dtype = dtype_map.get(dtype_name, torch.bfloat16)
        device = self.config.get("device", "cuda:0")

        self._hf_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype if dtype != "auto" else None,
            device_map=device,
        )
        self._hf_model.eval()
        self._hf_device = device

        sampling_cfg = dict(self.config.get("sampling") or {})
        self._hf_max_new_tokens = int(sampling_cfg.get("max_tokens", 256))
        self._hf_temperature = float(sampling_cfg.get("temperature", 0.0))

    def generate(self, requests: Iterable[VLMRequest]) -> list[VLMResponse]:
        reqs = list(requests)
        if not reqs:
            return []
        if self._backend == "hf":
            return self._generate_hf(reqs)
        return self._generate_vllm(reqs)

    def _generate_vllm(self, reqs: list[VLMRequest]) -> list[VLMResponse]:
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

    def _generate_hf(self, reqs: list[VLMRequest]) -> list[VLMResponse]:
        import torch
        from qwen_vl_utils import process_vision_info

        responses: list[VLMResponse] = []
        do_sample = self._hf_temperature > 0.0
        gen_kwargs: dict[str, Any] = {"max_new_tokens": self._hf_max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = self._hf_temperature

        for r in reqs:
            if len(r.images) != 1:
                raise ValueError(
                    f"LLaVA-OneVision-1.5 expects exactly 1 image per request, got {len(r.images)}"
                )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": r.images[0]},
                        {"type": "text", "text": r.prompt},
                    ],
                }
            ]
            prompt = self._hf_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._hf_processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            moved = {k: (v.to(self._hf_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

            with torch.inference_mode():
                out = self._hf_model.generate(**moved, **gen_kwargs)
            in_len = moved["input_ids"].shape[-1]
            gen_tokens = out[0, in_len:]
            text = self._hf_processor.decode(gen_tokens, skip_special_tokens=True).strip()
            responses.append(VLMResponse(text=text, metadata={"backend": "hf"}))
        return responses

    def shutdown(self) -> None:
        if getattr(self, "_backend", "vllm") == "hf":
            self._hf_model = None
            self._hf_processor = None
        else:
            self._llm = None
