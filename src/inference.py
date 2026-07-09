"""
Inference with a loaded LoRA adapter.
Loads the base model, applies a saved adapter, and generates predictions.
"""

import torch
import logging
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_adapter_for_inference(
    base_model_name: str,
    adapter_path: str,
    torch_dtype: str = "float16"
):
    """Load a base model with a trained LoRA adapter."""
    dtype = torch.float16 if torch_dtype == "float16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )

    logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def predict_sentiment(
    model,
    tokenizer,
    review: str,
    device: str,
    temperature: float = 0.1,
    max_new_tokens: int = 10
) -> str:
    """Run inference on a single review."""
    prompt = f"Review: {review}\nSentiment:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sentiment = result.replace(prompt, "").strip().split("\n")[0]
    return sentiment
