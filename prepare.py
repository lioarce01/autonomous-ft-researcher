"""
Download and prepare all data for the IFEval fine-tuning researcher.

Downloads:
  1. IFEval prompts  → data/ifeval_prompts.jsonl  (541 prompts)
  2. Alpaca 52k      → data/train_alpaca.jsonl
  3. UltraFeedback   → data/train_ultrafeedback.jsonl (10k high-quality subset, score ≥ 4.5)

The model weights (Qwen3.5-2B/) are expected to already exist in the project root.
Run this script once before starting experiments.
"""
import os
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ── 1. IFEval prompts ────────────────────────────────────────────────────────

def download_ifeval():
    out_path = os.path.join(DATA_DIR, "ifeval_prompts.jsonl")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return
    print("Downloading IFEval prompts from HuggingFace (google/ifeval)...")
    from datasets import load_dataset
    ds = load_dataset("google/ifeval", split="train")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")
    print(f"  Saved {len(ds)} prompts → {out_path}")


# ── 2. Alpaca 52k ────────────────────────────────────────────────────────────

def download_alpaca():
    out_path = os.path.join(DATA_DIR, "train_alpaca.jsonl")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return
    print("Downloading Alpaca dataset (tatsu-lab/alpaca)...")
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            instruction = row["instruction"]
            if row.get("input"):
                instruction = instruction + "\n\n" + row["input"]
            output = row["output"]
            if instruction and output:
                f.write(json.dumps({"instruction": instruction, "output": output}) + "\n")
    print(f"  Saved {len(ds)} examples → {out_path}")


# ── 3. UltraFeedback (10k high-quality subset) ───────────────────────────────

def download_ultrafeedback():
    out_path = os.path.join(DATA_DIR, "train_ultrafeedback.jsonl")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return
    print("Downloading UltraFeedback (openbmb/UltraFeedback), filtering score ≥ 4.5...")
    from datasets import load_dataset
    ds = load_dataset("openbmb/UltraFeedback", split="train")

    examples = []
    for row in ds:
        instruction = row.get("instruction", "")
        completions = row.get("completions", [])
        for comp in completions:
            annotations = comp.get("annotations", {})
            # Average the scores across criteria
            scores = []
            for crit in annotations.values():
                try:
                    scores.append(float(crit.get("Rating", 0)))
                except (ValueError, TypeError):
                    pass
            if not scores:
                continue
            avg_score = sum(scores) / len(scores)
            if avg_score >= 4.5:
                output = comp.get("response", "")
                if instruction and output:
                    examples.append({"instruction": instruction, "output": output})
        if len(examples) >= 10_000:
            break

    examples = examples[:10_000]
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved {len(examples)} high-quality examples → {out_path}")


if __name__ == "__main__":
    download_ifeval()
    download_alpaca()
    download_ultrafeedback()
    print("\nAll data ready. Run: uv run python evaluate.py --no-adapter")
