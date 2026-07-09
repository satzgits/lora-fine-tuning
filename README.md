# LoRA Fine-Tuning Pipeline

Fine-tunes large language models using Low-Rank Adaptation (LoRA) — a parameter-efficient technique that trains only 0.1-1% of the total parameters. Works with HuggingFace models on consumer GPUs (8GB VRAM is sufficient for 1.5B parameter models).

## Pipeline

```
Load Base Model → Apply LoRA Adapters → Train on Custom Data → Save Adapter (~10MB) → Load & Inference
```

Each stage is an independent module — swap models, datasets, or LoRA configs without touching the pipeline code.

## Project Structure

```
lora-fine-tuning/
├── config/
│   └── training_config.yaml      # All hyperparameters in one place
├── src/
│   ├── data.py                   # Dataset creation + tokenization
│   ├── model_setup.py            # Base model loading + LoRA injection
│   ├── train.py                  # HuggingFace Trainer wrapper
│   ├── inference.py              # Adapter reload + generation
│   └── utils.py                  # Logging, YAML loading, saving
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py          # Config + data tests
├── main.py                       # Orchestrator: load → train → save → infer
├── Dockerfile
├── Makefile                      # install, train, test, docker
├── pyproject.toml                # Project metadata + dep management
├── requirements.txt
└── README.md
```

## Quick Start

```bash
make install      # pip install -r requirements.txt
make train        # python main.py
make test         # pytest tests/ -v
```

## Configuration

All parameters live in `config/training_config.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  torch_dtype: "float16"

lora:
  r: 8                # rank — higher = more capacity, more VRAM
  alpha: 16           # scaling factor
  target_modules:     # which layers get adapters
    - "q_proj"
    - "v_proj"

training:
  epochs: 3
  batch_size: 2
  learning_rate: 2.0e-4

data:
  num_samples: 12
```

Change any value without editing Python code.

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

## Pipeline Stages

| Stage | Module | What Happens |
|-------|--------|-------------|
| 1. Load base model | `model_setup.py` | Downloads Qwen2.5-1.5B-Instruct from HuggingFace (3GB in FP16) |
| 2. Apply LoRA | `model_setup.py` | Injects adapters on query, key, value, output projections (rank=8) |
| 3. Prepare data | `data.py` | Formats 12 sentiment-labeled reviews as instruction examples |
| 4. Train | `train.py` | Fine-tunes for 3 epochs with FP16 mixed precision |
| 5. Save adapter | `utils.py` | Writes adapter weights to `./lora-adapter/` (~10MB) |
| 6. Load + infer | `inference.py` | Loads adapter on fresh base model, runs inference on test reviews |

## Docker

```bash
make docker       # docker build -t lora-fine-tuning .
docker run --gpus all lora-fine-tuning   # Requires NVIDIA Container Toolkit
```

For CPU-only testing:
```bash
make docker
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

- **Swap model:** Change `model.name` in `config/training_config.yaml`
- **Use your dataset:** Edit `src/data.py` → `create_sentiment_dataset()`
- **Adjust LoRA params:** Edit `config.lora` in the YAML file
- **More epochs/larger data:** Edit `config.training` and `config.data`

## Dependencies

- PyTorch (CUDA if GPU available)
- HuggingFace Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets + Accelerate
- PyYAML
