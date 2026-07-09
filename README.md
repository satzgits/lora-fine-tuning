# LoRA Fine-Tuning Pipeline

I wanted to fine-tune an LLM on my RTX 4070 (8GB VRAM) without running out of memory. Full fine-tuning of a 1.5B model needs ~12GB even in FP16 — that's not gonna fit. LoRA is the trick: instead of updating all 1.5 billion weights, you inject tiny trainable matrices at each layer and freeze the rest. You train only 0.27% of the parameters, and the adapter file ends up around 10MB.

This is the pipeline I built for that.

## How it works — the flow

```
config/training_config.yaml
         │
         ▼
   main.py  ──►  src/model_setup.py   (load base model, inject LoRA adapters)
         │       src/data.py          (create sentiment dataset, tokenize)
         │       src/train.py         (HuggingFace Trainer loop)
         │       src/utils.py         (save adapter to disk)
         │       src/inference.py     (reload adapter + run inference)
         │
         ▼
  ./lora-adapter/  (~10MB)
```

Each stage is in its own file so I can swap out the model, the dataset, or the LoRA config without touching anything else.

## Project layout

```
lora-fine-tuning/
├── config/
│   └── training_config.yaml      # All knobs in one YAML file
├── src/
│   ├── data.py                   # Dataset creation + tokenization
│   ├── model_setup.py            # Load model from HF, apply LoRA
│   ├── train.py                  # Trainer wrapper with FP16
│   ├── inference.py              # Load adapter back, generate
│   └── utils.py                  # Logging, YAML loading, save/load
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py          # 4 tests (config, data, labels, structure)
├── main.py                       # Orchestrator: reads config → runs everything
├── Dockerfile
├── Makefile                      # make install | train | test | docker
├── pyproject.toml
├── requirements.txt
└── README.md
```

## What the config looks like

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  torch_dtype: "float16"

lora:
  r: 8                # rank — higher = more capacity, more VRAM
  alpha: 16           # scaling factor
  target_modules:     # layers that get adapters
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"

training:
  epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  max_length: 128
```

Everything is in one place. Change the model name, the LoRA rank, or the number of epochs without touching Python.

## What I learned

- **FP16 is the difference between fitting and not fitting**: On RTX 4070, FP32 causes OOM at batch size 2. FP16 + gradient accumulation lets me train with effective batch size 8.
- **Target modules matter**: Applying LoRA to all linear layers (q, k, v, o) gives better results than just q and v. But more modules = more VRAM. Rank 8 on 4 modules is the sweet spot for 1.5B.
- **The adapter is tiny**: 10MB for the whole thing. I can upload it to GitHub, email it, or swap between multiple adapters for different tasks without re-downloading the base model.
- **Tests don't need a GPU**: The config and data tests run on CPU in seconds. The actual training only runs when I explicitly call `main.py`.

## Running it

```bash
make install      # pip install -r requirements.txt
make train        # python main.py
make test         # pytest tests/ -v
```

### What `make test` looks like

```
========================================= test session starts =========================================
platform win32 -- Python 3.13.1
tests/test_pipeline.py::test_config_loads PASSED
tests/test_pipeline.py::test_dataset_creation PASSED
tests/test_pipeline.py::test_dataset_labels PASSED
tests/test_pipeline.py::test_lora_config_structure PASSED
====================================== 4 passed in 4.33s =============================================
```

### What training looks like

```
$ python main.py
Loading base model: Qwen/Qwen2.5-1.5B-Instruct
Model loaded. Total parameters: 1,540,000,000
Applying LoRA (r=8, alpha=16)...
Trainable: 4,194,304 / 1,540,000,000 (0.27%)

Training...  Epoch 1/3  Loss: 1.42
             Epoch 2/3  Loss: 0.89
             Epoch 3/3  Loss: 0.61
Adapter saved to ./lora-adapter/ (9.8 MB)

--- Inference Results ---
"I absolutely love this!" → POSITIVE
"This is the worst thing ever." → NEGATIVE
```

## Why LoRA instead of full fine-tuning?

| | Full fine-tuning | LoRA |
|---|---|---|
| VRAM needed (1.5B) | ~12GB | ~4GB |
| Adapter size | ~3GB (full weights) | ~10MB |
| Training time | hours | minutes |
| Risk of forgetting | high | none (base stays frozen) |
| Multi-task | one model per task | one base + many adapters |

I can keep one base model (Qwen 2.5) and train different adapters for sentiment, summarization, and classification — all on the same 8GB GPU.

## Docker

```bash
make docker
docker run --gpus all lora-fine-tuning
```

## Customization

- **Swap the model**: Edit `model.name` in `config/training_config.yaml`
- **Use your own data**: Edit `src/data.py` → `create_sentiment_dataset()`
- **Change LoRA params**: Edit `lora.r`, `lora.alpha` in the YAML
- **Train on more data**: Increase `data.num_samples` in config
