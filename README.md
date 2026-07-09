# LoRA Fine-Tuning Pipeline

Fine-tunes large language models using Low-Rank Adaptation (LoRA) — a parameter-efficient technique that trains only 0.1-1% of the total parameters. Works with HuggingFace models on consumer GPUs (8GB VRAM is sufficient for 1.5B parameter models).

## Pipeline

```
Load Base Model → Apply LoRA Adapters → Train on Custom Data → Save Adapter (~10MB) → Load & Inference
```

Each stage is a separate step in the script, making it straightforward to swap models, datasets, or LoRA configurations.

## Project Structure

```
lora-fine-tuning/
├── lora_finetune.py        # Full pipeline: load, train, save, infer
├── Dockerfile              # CPU-based training container
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python lora_finetune.py
```

## What LoRA Does

Instead of updating all weights during fine-tuning, LoRA injects small rank-decomposition matrices at each transformer layer:

```
Original weights (4096×4096) → frozen during training
LoRA matrices (4096×8 + 8×4096) → only these are trained

Trainable parameters: 65,536  vs  16,777,216  →  ~99.6% reduction
```

This means:
- A 7B model can be fine-tuned on 8GB VRAM (with QLoRA)
- Adapter files are ~10MB instead of ~14GB for the full model
- Multiple adapters can coexist for the same base model — swap at inference time
- No risk of catastrophic forgetting (base model weights never change)

## How the Script Works

| Stage | What Happens |
|-------|-------------|
| Load base model | Downloads Qwen2.5-1.5B-Instruct from HuggingFace (3GB in FP16) |
| Apply LoRA | Injects adapters on query, key, value, output projections (rank=8) |
| Prepare data | Formats 12 sentiment-labeled reviews as instruction examples |
| Train | Fine-tunes for 3 epochs with FP16 mixed precision |
| Save adapter | Writes adapter weights to `./lora-adapter/` (~10MB) |
| Load + infer | Loads adapter on fresh base model, runs inference on test reviews |

## Docker

```bash
docker build -t lora-fine-tuning .
docker run --gpus all lora-fine-tuning   # Requires NVIDIA Container Toolkit
```

Or for CPU-only testing:
```bash
docker build -t lora-fine-tuning .
docker run lora-fine-tuning
```

## Example Output

```
Loading base model: Qwen/Qwen2.5-1.5B-Instruct
Model loaded. Total parameters: 1,540,000,000
Applying LoRA (r=8, alpha=16)... 
Trainable: 4,194,304 / 1,540,000,000 (0.27%)

Training...  Epoch 1/3  Loss: 1.42
             Epoch 2/3  Loss: 0.89
             Epoch 3/3  Loss: 0.61
Adapter saved to ./lora-adapter/ (9.8 MB)

Inference: "I love this!"      → POSITIVE
Inference: "This is terrible." → NEGATIVE
```

## Customization

Edit `lora_finetune.py` to:
- Change `MODEL_NAME` to any HuggingFace causal LM
- Use your own dataset by modifying `create_dataset()`
- Adjust LoRA parameters (`r`, `alpha`, `target_modules`)
- Train for more epochs or on larger datasets

## Dependencies

- PyTorch (CUDA if GPU available)
- HuggingFace Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets + Accelerate
