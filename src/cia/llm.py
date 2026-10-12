"""Open-source text generation with Hugging Face Transformers, with an optional LoRA adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Generator(Protocol):
    def generate(self, messages: list[dict]) -> str: ...


def pick_device() -> str:
    """NVIDIA GPU if present, else the Apple Silicon GPU (MPS), else CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFGenerator:
    """Chat-style generation from any instruction-tuned causal LM on the Hugging Face Hub.

    Pass ``adapter_path`` to load a LoRA adapter produced by ``cia.finetune.train_lora``.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: str | Path | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = {
            "cuda": torch.bfloat16,
            "mps": torch.float16,  # Apple Silicon GPU: half precision halves memory use
            "cpu": torch.float32,
        }[self.device]
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, messages: list[dict]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        sampling = self.temperature > 0
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=sampling,
                temperature=self.temperature if sampling else None,
                top_p=0.9 if sampling else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
