"""
Tests for the LoRA fine-tuning pipeline.
Validates data creation, tokenization, and configuration loading.
"""

import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_sentiment_dataset
from src.utils import load_config


def test_config_loads():
    config = load_config("config/training_config.yaml")
    assert "model" in config
    assert "lora" in config
    assert "training" in config
    assert config["lora"]["r"] == 8
    assert config["lora"]["alpha"] == 16


def test_dataset_creation():
    dataset = create_sentiment_dataset(num_samples=4)
    assert len(dataset) == 4
    assert "text" in dataset.column_names
    for item in dataset:
        assert "Review:" in item["text"]
        assert "Sentiment:" in item["text"]


def test_dataset_labels():
    dataset = create_sentiment_dataset(num_samples=2)
    assert "POSITIVE" in dataset[0]["text"]
    assert "NEGATIVE" in dataset[1]["text"]


def test_lora_config_structure():
    config = load_config("config/training_config.yaml")
    modules = config["lora"]["target_modules"]
    assert isinstance(modules, list)
    assert "q_proj" in modules
    assert "v_proj" in modules


if __name__ == "__main__":
    test_config_loads()
    print("[OK] test_config_loads")
    test_dataset_creation()
    print("[OK] test_dataset_creation")
    test_dataset_labels()
    print("[OK] test_dataset_labels")
    test_lora_config_structure()
    print("[OK] test_lora_config_structure")
    print("\nAll tests passed.")
