# LoRA Fine-Tuning Pipeline

> **What:** Parameter-efficient fine-tuning of LLMs using LoRA (Low-Rank Adaptation)  
> **Stack:** Python, PyTorch, HuggingFace Transformers, PEFT, Datasets  
> **Target Role:** AI/ML Engineer (Kisai Technologies)  
> **GitHub:** https://github.com/satzgits/lora-fine-tuning

---

## What This Project Does

LoRA (Low-Rank Adaptation) is the standard way companies fine-tune LLMs in production. Instead of updating all 7 billion parameters (expensive, slow), LoRA injects small trainable matrices that are **0.1-1% of the model size**.

This pipeline demonstrates the full LoRA workflow:

```
Raw Data → Load Base Model → Apply LoRA → Train → Save Adapter → Load + Inference
```

## What You'll Learn

| Skill | Why It Matters for Kisai |
|-------|--------------------------|
| **LoRA / PEFT** | Kisai fine-tunes private models for enterprise customers |
| **HuggingFace Transformers** | Industry standard for loading/training LLMs |
| **Model checkpointing** | Saving/loading adapter weights — not full models |
| **GPU memory optimization** | Training 1.5B+ models on 8GB VRAM |
| **Training loop** | Loss tracking, evaluation, checkpointing |

## Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Load    │───►│  Apply   │───►│  Train   │───►│  Save    │───►│  Load +  │
│  Base    │    │  LoRA    │    │  Adapter │    │  Adapter │    │  Infer   │
│  Model   │    │          │    │          │    │  (10MB)  │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
    │               │               │               │               │
    ▼               ▼               ▼               ▼               ▼
Qwen2.5-1.5B   lora_config =     Trains only    adapter.safeten- Model loads
(3GB VRAM)     LoraConfig(       LoRA weights    sors = 10MB    adapter from
               r=8, lora_alpha=  (not full                        checkpoint
               16, target_modules model)                          + generates
               =["q_proj","v_proj"])                              text
```

## Files

| File | Purpose |
|------|---------|
| `lora_finetune.py` | Main pipeline — load, train, save, inference |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## How to Run

```bash
pip install -r requirements.txt
python lora_finetune.py
```

**Expected output:**
```
Loading base model...
Applying LoRA configuration...
Training... (runs for 3 steps on synthetic data)
Loss: 1.234
Saving adapter to ./lora-adapter...
Loading adapter back...
Inference: "I love this!" → POSITIVE (confidence: 0.92)
```

## Interview Script

> "I built a LoRA fine-tuning pipeline that demonstrates how to efficiently adapt LLMs to custom tasks. The pipeline loads a pre-trained model, applies LoRA adapters using HuggingFace's PEFT library — which reduces trainable parameters from billions to millions — trains on a custom dataset, saves the 10MB adapter file, and loads it back for inference. This is exactly how companies like Kisai fine-tune models for enterprise customers without needing expensive hardware or retraining the entire model."

## Relevance to Kisai

Kisai's platform offers "Private models trained/tuned on proprietary enterprise data." That's LoRA. Your ability to demonstrate this pipeline shows you understand their core product offering.
