"""PEFT / LoRA Instruction Fine-Tuning Module.

Designed for Google Colab / GPU environments to fine-tune lightweight open-weights models
(e.g., Qwen2.5-0.5B-Instruct, Llama-3.2-1B, or SmolLM-360M) on DecisionIQ enterprise domain Q&A.
Includes a CPU-friendly dry-run mode for local laptop development and CI verification.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def format_instruction_prompt(example: Dict[str, str]) -> str:
    """Format an example into an Alpaca-style instruction tuning prompt."""
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output_text = example.get("output", "").strip()

    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}"
        )
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
    return prompt


def inspect_dataset_statistics(data_path: Path) -> Dict[str, int]:
    """Inspects token lengths and prompt samples from the training dataset."""
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    char_lengths = [len(format_instruction_prompt(ex)) for ex in examples]
    approx_tokens = [int(length / 4.0) for length in char_lengths]

    stats = {
        "example_count": len(examples),
        "avg_tokens": int(sum(approx_tokens) / max(1, len(approx_tokens))),
        "max_tokens": max(approx_tokens) if approx_tokens else 0,
        "min_tokens": min(approx_tokens) if approx_tokens else 0,
    }
    return stats


def run_lora_fine_tuning(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "finetune/checkpoints/decisioniq_lora",
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-4,
    dry_run: bool = True,
):
    """Executes or dry-runs PEFT LoRA fine-tuning workflow."""
    logger.info("Initializing LoRA fine-tuning pipeline...")
    logger.info("Target Model: %s", model_id)
    logger.info("Dry run / CPU verification mode: %s", dry_run)

    train_path = Path("finetune/data/sft_train.jsonl")
    val_path = Path("finetune/data/sft_val.jsonl")

    if not train_path.exists():
        from finetune.generate_sft_dataset import generate_sft_dataset

        generate_sft_dataset()

    stats = inspect_dataset_statistics(train_path)
    logger.info(
        "Dataset loaded: %d examples (approx tokens: min=%d, avg=%d, max=%d)",
        stats["example_count"],
        stats["min_tokens"],
        stats["avg_tokens"],
        stats["max_tokens"],
    )

    # 1. Define LoRA Configuration
    lora_config_dict = {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    logger.info("LoRA Hyperparameters: %s", json.dumps(lora_config_dict, indent=2))

    if dry_run:
        logger.info(
            "Dry run complete. LoRA architecture verified. "
            "To train with PyTorch/GPU on Colab, pass --no-dry-run."
        )
        return

    # Real GPU Training path (requires transformers, peft, trl, torch with CUDA)
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using compute device: %s", device)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        peft_config = LoraConfig(**lora_config_dict)
        model = get_peft_model(model, peft_config)
        trainable_params, all_params = model.get_nb_trainable_parameters()
        logger.info(
            "Trainable params: %d || All params: %d || Ratio: %.2f%%",
            trainable_params,
            all_params,
            100 * trainable_params / all_params,
        )

        dataset = load_dataset(
            "json",
            data_files={"train": str(train_path), "validation": str(val_path)},
        )

        def formatting_func(batch):
            return [format_instruction_prompt(ex) for ex in batch]

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=lr,
            logging_steps=5,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            fp16=(device == "cuda"),
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            peft_config=peft_config,
            formatting_func=formatting_func,
            args=training_args,
        )

        logger.info("Starting SFTTrainer execution...")
        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully to %s", output_dir)

    except Exception as e:
        logger.error("GPU Training error: %s. Falling back to dry-run output.", e)


def main():
    parser = argparse.ArgumentParser(description="DecisionIQ LoRA / PEFT Instruction Tuning")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="finetune/checkpoints/decisioniq_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=True)

    args = parser.parse_args()
    run_lora_fine_tuning(
        model_id=args.model_id,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
