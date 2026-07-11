"""
Evaluate a trained LoRA adapter on test data.
Loads a saved adapter from disk and runs inference on sample reviews.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.inference import load_adapter_for_inference, predict_sentiment
from src.utils import setup_logging

setup_logging()


def load_config():
    with open("config/training_config.yaml") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    model_cfg = config["model"]
    infer_cfg = config["inference"]

    print("Loading adapter from ./lora-adapter/")
    model, tokenizer, device = load_adapter_for_inference(
        model_cfg["name"],
        "./lora-adapter",
        model_cfg["torch_dtype"]
    )

    test_reviews = [
        "I absolutely love this product!",
        "This is the worst thing ever.",
        "Pretty good, would recommend.",
        "Terrible quality, broke immediately.",
        "Amazing service and fast delivery!",
        "Not worth the money at all.",
        "Works perfectly, very satisfied.",
        "Horrible experience, avoid this.",
        "It's okay, nothing special.",
        "Best purchase I've made this year!",
    ]

    print("\n--- Evaluation Results ---")
    for review in test_reviews:
        sentiment = predict_sentiment(
            model, tokenizer, review, device,
            temperature=infer_cfg["temperature"],
            max_new_tokens=infer_cfg["max_new_tokens"]
        )
        print(f"  \"{review}\"")
        print(f"  → {sentiment}\n")


if __name__ == "__main__":
    main()
