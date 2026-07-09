"""
Dataset preparation for LoRA fine-tuning.
Creates instruction-formatted sentiment examples from raw text.
"""

from datasets import Dataset


def create_sentiment_dataset(num_samples: int = 12) -> Dataset:
    """Create a labelled sentiment dataset for instruction fine-tuning."""
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

    formatted = []
    for text, label in zip(texts[:num_samples], labels[:num_samples]):
        sentiment = "POSITIVE" if label == 1 else "NEGATIVE"
        formatted.append(f"Review: {text}\nSentiment: {sentiment}")

    return Dataset.from_dict({"text": formatted})


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 128):
    """Tokenize dataset for causal LM training."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
