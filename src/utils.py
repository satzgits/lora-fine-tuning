"""
Utility functions — logging setup, YAML loading, adapter saving.
"""

import yaml
import logging
from pathlib import Path


def setup_logging(level: str = "INFO"):
    """Configure logging format and level."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )


def load_config(path: str = "config/training_config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_adapter(model, tokenizer, path: str = "./lora-adapter"):
    """Save LoRA adapter weights and tokenizer."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    logging.getLogger(__name__).info(f"Adapter saved to {path}/")
