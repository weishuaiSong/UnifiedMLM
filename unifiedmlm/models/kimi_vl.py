"""Kimi-VL-A3B-Instruct (Moonshot MoE 16B / 3B-active + MoonViT) over vLLM **或** HF.

HF: moonshotai/Kimi-VL-A3B-Instruct

vLLM 0.11.2 原生支持 arch KimiVLForConditionalGeneration，但 16B total weights ~32G
在单张 24G 卡上 OOM。默认走 HF backend + `device_map="auto"` 跨多卡 split。

Backend 选项（config["backend"]）：
  - "vllm": vLLM 后端（需多卡 TP 或更大显存）
  - "hf"  (default): transformers AutoModelForCausalLM + device_map="auto"

Chat prompt (from chat_template.jinja, image variant):
    <|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|>
    <|im_user|>user<|im_middle|><|media_start|>image<|media_content|><|media_pad|><|media_end|>{question}<|im_end|>
    <|im_assistant|>assistant<|im_middle|>
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import BaseVLMModel, VLMRequest, VLMResponse
from .registry import register_model


KIMI_VL_PROMPT = (
    "<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|>"
    "<|im_user|>user<|im_middle|>"
    "<|media_start|>image<|media_content|><|media_pad|><|media_end|>"
    "{question}<|im_end|>"
    "<|im_assistant|>assistant<|im_middle|>"
)


@register_model("kimi-vl-a3b")
class KimiVL_A3B(BaseVLMModel):
    """Kimi-VL-A3B-Instruct wrapper, vLLM 或 HF.

    Config keys:
      backend:           "vllm" | "hf"  (default "hf"，16B MoE 单卡装不下)
      model_path:        HF repo / local path (default: moonshotai/Kimi-VL-A3B-Instruct)
      tensor_parallel:   vllm only,  int, default 1
      max_model_len:     vllm only,  int, default 8192
      gpu_memory_util:   vllm only,  float, default 0.9
      dtype:             "bfloat16" | "float16" | "auto"
      device_map:        hf  only,   default "auto"
      sampling:          dict (temperature, max_tokens)
    """

    DEFAULT_MODEL_PATH = "moonshotai/Kimi-VL-A3B-Instruct"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._backend = str(self.config.get("backend", "hf")).lower()
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
        # Kimi-VL remote code imports `PytorchGELUTanh` from transformers.activations
        # which was renamed to `GELUTanh` in newer transformers. Alias before import.
        import transformers.activations as _act
        if not hasattr(_act, "PytorchGELUTanh") and hasattr(_act, "GELUTanh"):
            _act.PytorchGELUTanh = _act.GELUTanh
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_path = self.config.get("model_path", self.DEFAULT_MODEL_PATH)
        dtype_name = str(self.config.get("dtype", "bfloat16")).lower()
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(dtype_name, torch.bfloat16)
        device_map = self.config.get("device_map", "auto")

        self._hf_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device_map,
        )
        self._hf_model.eval()
        first_param = next(self._hf_model.parameters())
        self._hf_input_device = first_param.device
        self._hf_dtype = first_param.dtype

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
                    f"Kimi-VL expects exactly 1 image per request, got {len(r.images)}"
                )
            vllm_inputs.append(
                {
                    "prompt": KIMI_VL_PROMPT.format(question=r.prompt),
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

        responses: list[VLMResponse] = []
        do_sample = self._hf_temperature > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self._hf_max_new_tokens,
            "do_sample": do_sample,
            "use_cache": False,  # Kimi remote code uses old DynamicCache.seen_tokens API
        }
        if do_sample:
            gen_kwargs["temperature"] = self._hf_temperature

        for r in reqs:
            if len(r.images) != 1:
                raise ValueError(
                    f"Kimi-VL expects exactly 1 image per request, got {len(r.images)}"
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
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = self._hf_processor(text=prompt, images=[r.images[0]], return_tensors="pt")
            moved: dict[str, Any] = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    if v.is_floating_point():
                        moved[k] = v.to(self._hf_input_device).to(self._hf_dtype)
                    else:
                        moved[k] = v.to(self._hf_input_device)
                else:
                    moved[k] = v

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
