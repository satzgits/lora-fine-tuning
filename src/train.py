"""
Training loop for LoRA fine-tuning using HuggingFace Trainer.
"""

import torch
import logging
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

logger = logging.getLogger(__name__)


def create_trainer(model, tokenizer, train_dataset, config: dict):
    """Configure and return a HuggingFace Trainer for LoRA fine-tuning."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = config["fp16"] and device == "cuda"

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["epochs"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        fp16=use_fp16,
        learning_rate=config["learning_rate"],
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    return trainer


def run_training(model, tokenizer, dataset, training_config: dict):
    """Run the training loop."""
    trainer = create_trainer(model, tokenizer, dataset, training_config)
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")
    return trainer
