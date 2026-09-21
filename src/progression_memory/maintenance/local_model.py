from __future__ import annotations

from functools import lru_cache


class TransformersLanguageModel:
    """Lazy local HF backend; optimized for short deterministic classification/state updates."""

    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM3-3B", *,
                 max_new_tokens: int = 160, quantization: str = "none") -> None:
        if quantization not in {"none", "quanto-int8"}:
            raise ValueError(f"unsupported quantization: {quantization}")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.quantization = quantization

    @property
    def runtime(self):
        return _load(self.model_name, self.quantization)

    def generate(self, prompt: str) -> str:
        return self.generate_messages([{"role": "user", "content": prompt}])

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict | None = None,
    ) -> str:
        tokenizer, model = self.runtime
        messages = _prepare_messages(self.model_name, messages)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        generation = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if json_schema is not None:
            try:
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import (
                    build_transformers_prefix_allowed_tokens_fn,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Structured generation requires the 'slm' extra: pip install -e '.[slm]'"
                ) from exc
            generation["prefix_allowed_tokens_fn"] = build_transformers_prefix_allowed_tokens_fn(
                tokenizer, JsonSchemaParser(json_schema)
            )
        output = model.generate(**inputs, **generation)
        return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _prepare_messages(model_name: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Follow the distilled R1 prompt contract while preserving few-shot turns."""
    if not model_name.casefold().startswith("deepseek-ai/deepseek-r1-distill-"):
        return messages
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    prepared = [dict(m) for m in messages if m["role"] != "system"]
    if system and prepared:
        prepared[0]["content"] = f"{system}\n\n{prepared[0]['content']}"
    return prepared


@lru_cache(maxsize=2)
def _load(model_name: str, quantization: str = "none"):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the local writer with: pip install -e '.[slm]'") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    options = {"torch_dtype": "auto", "device_map": "auto"}
    if quantization == "quanto-int8":
        try:
            from transformers import QuantoConfig
            options["quantization_config"] = QuantoConfig(weights="int8")
        except ImportError as exc:
            raise RuntimeError(
                "Quanto int8 requires the 'quantized' extra: "
                "pip install -e '.[slm,quantized]'"
            ) from exc
    model = AutoModelForCausalLM.from_pretrained(model_name, **options)
    model.eval()
    return tokenizer, model


def clear_model_cache() -> None:
    """Release cached writer models between bounded benchmark runs."""
    _load.cache_clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
