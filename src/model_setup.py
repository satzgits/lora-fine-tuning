"""
Model loading and LoRA adapter injection.
"""

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


def load_base_model(model_name: str, torch_dtype: str, device_map: str = "auto"):
    """Load a pre-trained model and tokenizer from HuggingFace."""
    dtype = torch.float16 if torch_dtype == "float16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params:,} parameters")
    return model, tokenizer


def apply_lora(model, lora_config: dict) -> tuple:
    """Inject LoRA adapters into the model."""
    config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["dropout"],
        bias=lora_config["bias"],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    logger.info(f"LoRA applied: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")

    return model
