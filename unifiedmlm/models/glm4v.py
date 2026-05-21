"""GLM-4V-9B (智谱 ChatGLM4 + EVA-CLIP-E) over vLLM **或** HF.

HF: THUDM/glm-4v-9b

vLLM 0.11.2 走 `hf_overrides: {"architectures": ["GLM4VForCausalLM"]}` 可加载，但
9B + vision 单张 24G 卡 OOM。默认走 HF backend + `device_map="auto"` 多卡 split。

Backend 选项（config["backend"]）：
  - "vllm": vLLM 后端（需 hf_overrides + 多卡 TP 或更大显存）
  - "hf"  (default): transformers AutoModelForCausalLM + device_map="auto"

⚠️ HF 模式注意（transformers 4.57 兼容性）：
   - ChatGLM remote code 用 `num_layers` 不是 `num_hidden_layers`，DynamicCache
     初始化时会报 AttributeError → wrapper 启动时打 num_hidden_layers alias。
   - GLM-4V 的 `get_masks` 访问 past_key_values[0][0].shape 但新版 cache 第 0
     项可能是 None → generate 时强制 `use_cache=False`。
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import BaseVLMModel, VLMRequest, VLMResponse
from .registry import register_model


# vLLM 文本端模板（image 由 vLLM 通过 multi_modal_data 自动注入，不需要占位符）。
GLM4V_PROMPT = "[gMASK]<sop><|user|>\n{question}<|assistant|>"


@register_model("glm-4v-9b")
class GLM4V_9B(BaseVLMModel):
    """GLM-4V-9B wrapper, vLLM 或 HF.

    Config keys:
      backend:         "vllm" | "hf"  (default "hf"，单卡 OOM)
      model_path:      HF repo / local path (default: THUDM/glm-4v-9b)
      tensor_parallel: vllm only,  int, default 1
      max_model_len:   vllm only,  int, default 4096
      gpu_memory_util: vllm only,  float, default 0.9
      dtype:           "bfloat16" | "float16" | "auto"
      device_map:      hf  only,   default "auto"
      sampling:        dict (temperature, max_tokens)
    """

    DEFAULT_MODEL_PATH = "THUDM/glm-4v-9b"

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
            max_model_len=int(self.config.get("max_model_len", 4096)),
            gpu_memory_utilization=float(self.config.get("gpu_memory_util", 0.9)),
            dtype=self.config.get("dtype", "auto"),
            trust_remote_code=bool(self.config.get("trust_remote_code", True)),
            limit_mm_per_prompt={"image": 1},
        )
        # GLM-4V config.json 写 ChatGLMModel（text-only），vLLM 视觉路径要走 GLM4VForCausalLM
        hf_overrides = self.config.get("hf_overrides") or {"architectures": ["GLM4VForCausalLM"]}
        llm_kwargs["hf_overrides"] = hf_overrides
        self._llm = LLM(**llm_kwargs)

        sampling_cfg = dict(self.config.get("sampling") or {})
        sampling_cfg.setdefault("temperature", 0.0)
        sampling_cfg.setdefault("max_tokens", 256)
        self._sampling = SamplingParams(**sampling_cfg)

    def _init_hf(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = self.config.get("model_path", self.DEFAULT_MODEL_PATH)
        dtype_name = str(self.config.get("dtype", "bfloat16")).lower()
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(dtype_name, torch.bfloat16)
        device_map = self.config.get("device_map", "auto")

        self._hf_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device_map,
        )
        self._hf_model.eval()
        # Compat patch: transformers 4.57 的 DynamicCache 要 num_hidden_layers
        if not hasattr(self._hf_model.config, "num_hidden_layers") and hasattr(self._hf_model.config, "num_layers"):
            self._hf_model.config.num_hidden_layers = self._hf_model.config.num_layers

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
                    f"GLM-4V expects exactly 1 image per request, got {len(r.images)}"
                )
            vllm_inputs.append(
                {
                    "prompt": GLM4V_PROMPT.format(question=r.prompt),
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
            "use_cache": False,  # 见文件顶部 compat 说明
        }
        if do_sample:
            gen_kwargs["temperature"] = self._hf_temperature

        for r in reqs:
            if len(r.images) != 1:
                raise ValueError(
                    f"GLM-4V expects exactly 1 image per request, got {len(r.images)}"
                )
            # GLM-4V 的 apply_chat_template 直接接 "image" 字段（不是 OpenAI content list）
            inputs = self._hf_tokenizer.apply_chat_template(
                [{"role": "user", "image": r.images[0], "content": r.prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
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
            text = self._hf_tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            responses.append(VLMResponse(text=text, metadata={"backend": "hf"}))
        return responses

    def shutdown(self) -> None:
        if getattr(self, "_backend", "vllm") == "hf":
            self._hf_model = None
            self._hf_tokenizer = None
        else:
            self._llm = None
