"""
LoRA Fine-Tuning Pipeline
=========================
Trains a small LLM using Low-Rank Adaptation (LoRA).
Demonstrates production-ready parameter-efficient fine-tuning.

Pipeline: Load model → Apply LoRA → Train on sentiment data → Save adapter → Inference
"""

import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_dataset():
    """Create a tiny sentiment dataset for demonstration."""
    texts = [
        "I love this product! It's amazing.",
        "This is terrible, I hate it.",
        "Great quality, very satisfied.",
        "Worst purchase ever, completely disappointed.",
        "Fantastic service and fast delivery.",
        "Poor quality, broke in one day.",
        "Absolutely wonderful experience!",
        "Not worth the money at all.",
        "Best thing I've bought this year.",
        "Horrible customer support.",
        "Works perfectly, highly recommend!",
        "Very bad, do not buy.",
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    # Format as instruction-following examples
    formatted = []
    for text, label in zip(texts, labels):
        sentiment = "POSITIVE" if label == 1 else "NEGATIVE"
        formatted.append(f"Review: {text}\nSentiment: {sentiment}")

    return Dataset.from_dict({"text": formatted})


def main():
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

    logger.info(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    logger.info(f"Model loaded. Trainable params: {model.num_parameters():,}")

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA applied: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    # --- Prepare dataset ---
    dataset = create_dataset()

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # --- Training ---
    training_args = TrainingArguments(
        output_dir="./lora-checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_steps=1,
        save_strategy="no",
        fp16=device == "cuda",
        learning_rate=2e-4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    # --- Save adapter ---
    adapter_path = "./lora-adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"LoRA adapter saved to {adapter_path}/ (~10MB)")

    # --- Load adapter back for inference ---
    logger.info("Loading adapter back for inference...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    loaded_model = PeftModel.from_pretrained(base_model, adapter_path)
    loaded_model.to(device)
    logger.info("Adapter loaded. Running inference...")

    # --- Inference ---
    test_reviews = [
        "I absolutely love this!",
        "This is the worst thing ever.",
    ]

    for review in test_reviews:
        prompt = f"Review: {review}\nSentiment:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = loaded_model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sentiment = result.replace(prompt, "").strip().split("\n")[0]
        print(f"Inference: \"{review}\" → {sentiment}")


if __name__ == "__main__":
    main()
