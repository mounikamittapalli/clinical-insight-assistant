"""LoRA / PEFT fine-tuning of an open-source instruct model on the generated QA pairs.

    python -m cia.finetune.train_lora            # LoRA (GPU recommended)
    python -m cia.finetune.train_lora --qlora    # 4-bit QLoRA on a CUDA GPU

Only the assistant's answer tokens are trained on (prompt tokens are masked with -100),
so the model learns to answer and cite, not to reproduce the context.
The adapter (a few MB) is saved to ``paths.adapter_dir`` and loaded at inference with
``llm.load_adapter: true``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cia.config import Config, load_config


def tokenize_example(example: dict, tokenizer, max_length: int) -> dict:
    messages = example["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full = prompt + messages[-1]["content"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_length]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def train(cfg: Config, qlora: bool = False) -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    ft = cfg.finetune
    data_dir = Path(cfg.paths.finetune_dir)
    tokenizer = AutoTokenizer.from_pretrained(ft.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {}
    if qlora or ft.load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(ft.base_model, **load_kwargs)
    if "quantization_config" in load_kwargs:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=ft.lora_r,
        lora_alpha=ft.lora_alpha,
        lora_dropout=ft.lora_dropout,
        target_modules=list(ft.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )
    ds = ds.map(
        lambda ex: tokenize_example(ex, tokenizer, ft.max_length),
        remove_columns=ds["train"].column_names,
    )

    out_dir = Path(cfg.paths.adapter_dir)
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=ft.epochs,
        learning_rate=ft.learning_rate,
        per_device_train_batch_size=ft.batch_size,
        per_device_eval_batch_size=ft.batch_size,
        gradient_accumulation_steps=ft.grad_accum,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        report_to=["mlflow"] if _has_mlflow() else [],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    result = trainer.train()
    metrics = {**result.metrics, **trainer.evaluate()}
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _has_mlflow() -> bool:
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--qlora", action="store_true", help="4-bit QLoRA (CUDA GPU only)")
    args = parser.parse_args()
    print(json.dumps(train(load_config(args.config), qlora=args.qlora), indent=2))


if __name__ == "__main__":
    main()
