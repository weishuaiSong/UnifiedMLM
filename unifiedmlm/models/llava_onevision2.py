"""LLaVA-OneVision-2-8B-Instruct.

HF: lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct (EvolvingLMMs-Lab, 2026)
Backbone: Qwen3-8B + OneVision-Encoder（HEVC-style ViT，native-resolution）。
4B 版（Qwen3-4B-Instruct-2507 底座）官方标注 coming soon，发布后改
model_path 即可复用本 wrapper。

官方推理路径（模型卡 quickstart）：
    AutoModelForImageTextToText + AutoProcessor, trust_remote_code=True
    要求 transformers >= 5.7 —— 主 venv (trf 4.57) 跑不了，
    A100 上用 .venv-qwen35 (trf 5.10.2)。
图像直接以 PIL 传给 processor（messages 里只放 {"type": "image"} 占位），
不走 qwen_vl_utils（与 OV-1.5 不同）。
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import BaseVLMModel, VLMRequest, VLMResponse
from .registry import register_model


@register_model("llava-onevision-2-8b")
class LLaVAOneVision2(BaseVLMModel):
    """LLaVA-OneVision-2-8B-Instruct wrapper.

    Backend 选项（config["backend"]）：
      - "hf" (default):  transformers AutoModelForImageTextToText + trust_remote_code。
                         需要 transformers >= 5.7（.venv-qwen35 满足）。
      - "vllm":          vLLM 后端。0.11.2 肯定不识别该架构；.venv-qwen35 的新
                         vLLM 未验证 —— 报 arch not supported 就回 hf。

    Config keys:
      model_path:        HF repo or local path (default: lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct)
      backend:           "hf" | "vllm"
      dtype:             "bfloat16" | "float16" | "float32" | "auto"
      device:            cuda device / "auto" for hf backend (default cuda:0)
      tensor_parallel:   int, default 1                (vllm only)
      max_model_len:     int, default 8192             (vllm only)
      gpu_memory_util:   float, default 0.9            (vllm only)
      sampling:          dict (temperature, max_tokens)
    """

    DEFAULT_MODEL_PATH = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._backend = str(self.config.get("backend", "hf")).lower()
        if self._backend == "vllm":
            self._init_vllm()
        else:
            self._init_hf()

    def _init_hf(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_path = self.config.get("model_path", self.DEFAULT_MODEL_PATH)
        dtype_name = str(self.config.get("dtype", "bfloat16")).lower()
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "auto": "auto"}
        dtype = dtype_map.get(dtype_name, torch.bfloat16)
        device = self.config.get("device", "cuda:0")

        self._hf_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._hf_model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device,
        )
        self._hf_model.eval()
        self._hf_device = device

        sampling_cfg = dict(self.config.get("sampling") or {})
        self._hf_max_new_tokens = int(sampling_cfg.get("max_tokens", 256))
        self._hf_temperature = float(sampling_cfg.get("temperature", 0.0))

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

    def generate(self, requests: Iterable[VLMRequest]) -> list[VLMResponse]:
        reqs = list(requests)
        if not reqs:
            return []
        if self._backend == "vllm":
            return self._generate_vllm(reqs)
        return self._generate_hf(reqs)

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
                    f"LLaVA-OneVision-2 expects exactly 1 image per request, got {len(r.images)}"
                )
            # 模型卡 quickstart 口径：image 占位在前、问题文本在后，PIL 直接给 processor
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
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._hf_processor(
                text=[prompt],
                images=[r.images[0]],
                padding=True,
                return_tensors="pt",
            )
            moved = {k: (v.to(self._hf_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

            with torch.inference_mode():
                out = self._hf_model.generate(**moved, **gen_kwargs)
            in_len = moved["input_ids"].shape[-1]
            gen_tokens = out[0, in_len:]
            text = self._hf_processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            responses.append(VLMResponse(text=text, metadata={"backend": "hf"}))
        return responses

    def _generate_vllm(self, reqs: list[VLMRequest]) -> list[VLMResponse]:
        vllm_inputs = []
        for r in reqs:
            if len(r.images) != 1:
                raise ValueError(
                    f"LLaVA-OneVision-2 expects exactly 1 image per request, got {len(r.images)}"
                )
            # vLLM 路径用 tokenizer 的 chat template 渲染（含 <|vision_start|> 占位）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": r.prompt},
                    ],
                }
            ]
            prompt = self._llm.get_tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
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
        if getattr(self, "_backend", "hf") == "vllm":
            self._llm = None
        else:
            self._hf_model = None
            self._hf_processor = None
