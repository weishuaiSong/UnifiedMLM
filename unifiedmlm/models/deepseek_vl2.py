"""DeepSeek-VL2-Small (DeepSeek-MoE 16B / 2.4B-active + dynamic-resolution ViT).

HF: deepseek-ai/deepseek-vl2-small
vLLM 0.11.2 原生支持 model_type=deepseek_vl_v2 (arch DeepseekVLV2ForCausalLM)。

⚠️ 当前两条加载路径都被 upstream 阻塞，待 fix：

  (1) vLLM TP=2 (当前 wrapper 走的路径)
        bf16 weights ~31G 单卡装不下；TP=2 触发 vLLM 0.11.2 的
        `loaded_weight.shape[output_dim] == self.org_vocab_size`
        AssertionError。要等 vLLM 上游修这个 vocab-size 加载 bug，或升 vLLM
        到合 fix 的版本（注意 0.21+ 是 cu13 wheel，与 driver 12.4 不兼容）。

  (2) HF transformers backend (尝试过，docstring 留 reference 不实装)
        装 `pip install --no-deps git+https://github.com/deepseek-ai/DeepSeek-VL2`
        + attrdict，需要叠 4 个 monkey-patch 让它在 transformers 4.57 上 import：
          - transformers.models.llama.modeling_llama.LlamaFlashAttention2 stub
          - DynamicCache.seen_tokens property
          - DynamicCache.get_max_length = get_max_cache_shape
          - DynamicCache.get_usable_length = get_seq_length
        + 注入 GenerationMixin 到 DeepseekV2ForCausalLM。
        最后还是卡在 RoPE 实现的 cos[position_ids] 形状 mismatch — deepseek_vl2
        的 RoPE 跟新版 transformers 完全不兼容，需要改源码才能继续。

待 vLLM 上游或 deepseek-vl2 仓库新 commit 修复后，把 backend: hf 实装上即可。

Chat prompt (vLLM 用):
    <|User|>: <image>
    {question}

    <|Assistant|>:
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import BaseVLMModel, VLMRequest, VLMResponse
from .registry import register_model


DEEPSEEK_VL2_PROMPT = "<|User|>: <image>\n{question}\n\n<|Assistant|>:"


@register_model("deepseek-vl2-small")
class DeepSeekVL2Small(BaseVLMModel):
    """DeepSeek-VL2-Small wrapper over vLLM.

    Config keys:
      model_path:      HF repo / local path (default: deepseek-ai/deepseek-vl2-small)
      tensor_parallel: int, default 1
      max_model_len:   int, default 4096  (DeepSeek-VL2 default ctx 短，节省 KV)
      gpu_memory_util: float, default 0.9
      dtype:           str, default "auto"
      sampling:        dict (temperature, max_tokens)
    """

    DEFAULT_MODEL_PATH = "deepseek-ai/deepseek-vl2-small"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
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
                    f"DeepSeek-VL2 expects exactly 1 image per request, got {len(r.images)}"
                )
            vllm_inputs.append(
                {
                    "prompt": DEEPSEEK_VL2_PROMPT.format(question=r.prompt),
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
