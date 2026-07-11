"""
Display information about a saved LoRA adapter.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os


def get_dir_size(path: str) -> str:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    if total < 1024:
        return f"{total} B"
    elif total < 1024 ** 2:
        return f"{total / 1024:.1f} KB"
    else:
        return f"{total / (1024 ** 2):.1f} MB"


def main():
    adapter_path = Path("./lora-adapter")

    if not adapter_path.exists():
        print("No adapter found at ./lora-adapter/")
        print("Run 'python main.py' first to train an adapter.")
        return

    files = list(adapter_path.iterdir())
    print(f"Adapter location: {adapter_path.resolve()}")
    print(f"Total size: {get_dir_size(str(adapter_path))}")
    print(f"Files: {len(files)}")
    print()

    for f in sorted(files):
        size = f.stat().st_size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 ** 2:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024 ** 2):.1f} MB"
        print(f"  {f.name:30s} {size_str:>8s}")

    # Show adapter config if adapter_config.json exists
    config_file = adapter_path / "adapter_config.json"
    if config_file.exists():
        import json
        with open(config_file) as f:
            cfg = json.load(f)
        print(f"\nLoRA rank (r): {cfg.get('r', 'N/A')}")
        print(f"LoRA alpha: {cfg.get('lora_alpha', 'N/A')}")
        print(f"Target modules: {cfg.get('target_modules', 'N/A')}")
        print(f"Base model: {cfg.get('base_model_name_or_path', 'N/A')}")


if __name__ == "__main__":
    main()
