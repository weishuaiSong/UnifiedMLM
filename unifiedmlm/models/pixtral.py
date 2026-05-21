"""Pixtral-12B (Mistral Nemo + native-resolution ViT) served by vLLM **or** HF.

HF: mistral-experimental/pixtral-12b   (mistralai/Pixtral-12B-2409 的免 gated reupload)

vLLM 0.11.2 注册的 arch 是 LlavaForConditionalGeneration（Pixtral 在 HF 上以 LLaVA
变体形式注册）。**但 weights 是 bf16 ~24G，单张 24G 卡装不下；TP=2 在当前环境
NCCL 初始化失败**，所以默认走 HF backend + `device_map="auto"` 让 weights 自动
split 到所有可见 GPU 上。

Backend 选项（config["backend"]）：
  - "vllm": vLLM 后端（要么够大的单卡，要么 TP=2 能跑通）
  - "hf"  (default): transformers AutoModelForImageTextToText + device_map="auto"

Chat prompt (from chat_template.json):
    <s>[INST][IMG]{question}[/INST]
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import BaseVLMModel, VLMRequest, VLMResponse
from .registry import register_model


PIXTRAL_PROMPT = "<s>[INST][IMG]{question}[/INST]"


@register_model("pixtral-12b")
class Pixtral12B(BaseVLMModel):
    """Pixtral-12B wrapper, vLLM 或 HF。

    Config keys:
      backend:           "vllm" | "hf"            (default "hf"，单卡装不下走 device_map=auto)
      model_path:        HF repo / local path (default: mistral-experimental/pixtral-12b)
      tensor_parallel:   vllm only,  int, default 1
      max_model_len:     vllm only,  int, default 8192
      gpu_memory_util:   vllm only,  float, default 0.9
      dtype:             "bfloat16" | "float16" | "auto" (HF backend 默认 bfloat16)
      device_map:        hf  only,   str|dict, default "auto" (跨多卡 split)
      sampling:          dict (temperature, max_tokens)
    """

    DEFAULT_MODEL_PATH = "mistral-experimental/pixtral-12b"

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
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_path = self.config.get("model_path", self.DEFAULT_MODEL_PATH)
        dtype_name = str(self.config.get("dtype", "bfloat16")).lower()
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(dtype_name, torch.bfloat16)
        device_map = self.config.get("device_map", "auto")

        self._hf_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._hf_model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device_map,
        )
        self._hf_model.eval()
        # 取第一个 layer device / dtype 作为 inputs target
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
                    f"Pixtral expects exactly 1 image per request, got {len(r.images)}"
                )
            vllm_inputs.append(
                {
                    "prompt": PIXTRAL_PROMPT.format(question=r.prompt),
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
        gen_kwargs: dict[str, Any] = {"max_new_tokens": self._hf_max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = self._hf_temperature

        for r in reqs:
            if len(r.images) != 1:
                raise ValueError(
                    f"Pixtral expects exactly 1 image per request, got {len(r.images)}"
                )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
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
