"""
LoRA Fine-Tuning Pipeline — Entry Point
========================================
Orchestrates the full pipeline: load model → apply LoRA → prepare data → train → save → inference.

Usage:
    python main.py                          # Run with default config
    python main.py --config custom.yaml     # Run with custom config
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, load_config, save_adapter
from src.data import create_sentiment_dataset, tokenize_dataset
from src.model_setup import load_base_model, apply_lora
from src.train import run_training
from src.inference import load_adapter_for_inference, predict_sentiment


def main(config_path: str = "config/training_config.yaml"):
    setup_logging()
    config = load_config(config_path)

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    infer_cfg = config["inference"]

    # Step 1: Load base model
    model, tokenizer = load_base_model(
        model_cfg["name"],
        model_cfg["torch_dtype"],
        model_cfg["device_map"]
    )

    # Step 2: Apply LoRA
    model = apply_lora(model, lora_cfg)

    # Step 3: Prepare dataset
    dataset = create_sentiment_dataset(num_samples=data_cfg["num_samples"])
    tokenized = tokenize_dataset(dataset, tokenizer, max_length=train_cfg["max_length"])

    # Step 4: Train
    run_training(model, tokenizer, tokenized, train_cfg)

    # Step 5: Save adapter
    save_adapter(model, tokenizer, path="./lora-adapter")

    # Step 6: Load adapter back and run inference
    ft_model, ft_tokenizer, device = load_adapter_for_inference(
        model_cfg["name"],
        "./lora-adapter",
        model_cfg["torch_dtype"]
    )

    test_reviews = [
        "I absolutely love this!",
        "This is the worst thing ever.",
    ]

    print("\n--- Inference Results ---")
    for review in test_reviews:
        sentiment = predict_sentiment(
            ft_model, ft_tokenizer, review, device,
            temperature=infer_cfg["temperature"],
            max_new_tokens=infer_cfg["max_new_tokens"]
        )
        print(f"  \"{review}\" → {sentiment}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning Pipeline")
    parser.add_argument("--config", default="config/training_config.yaml",
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    main(args.config)
