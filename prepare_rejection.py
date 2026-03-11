"""
Rejection sampling on IFEval prompts.

For each of the 541 IFEval prompts, generate N_SAMPLES responses with temperature
sampling, run the verifiers, and keep only responses where ALL constraints pass.

This gives the highest-quality training signal: every kept example is a (prompt, response)
pair that our evaluator would score as correct.

Output: data/train_ifeval_rejection.jsonl
Format: {"instruction": "<ifeval prompt>", "output": "<passing response>"}

Usage:
    uv run python prepare_rejection.py                                    # use 0.8B base model
    uv run python prepare_rejection.py --model-path data/models/Qwen3.5-2B  # use 2B model
    uv run python prepare_rejection.py --adapter                          # use current adapter in data/adapter_tmp/
"""
import argparse
import json
import os
import sys

# Force line-buffered stdout so prints appear immediately when redirected to file
sys.stdout.reconfigure(line_buffering=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL_PATH = os.path.join(ROOT, "data", "models", "Qwen3.5-0.8B")
ADAPTER_PATH = os.path.join(ROOT, "data", "adapter_tmp")
IFEVAL_PATH  = os.path.join(ROOT, "data", "ifeval_prompts.jsonl")
OUT_PATH     = os.path.join(ROOT, "data", "train_ifeval_rejection.jsonl")

N_SAMPLES       = 8     # responses generated per prompt
TEMPERATURE     = 0.8   # sampling temperature for diversity
MAX_NEW_TOKENS  = 700   # slightly more headroom than eval for word-count constraints
BATCH_SIZE      = 16    # different prompts per batch (same as evaluate.py)


def load_model(use_adapter: bool, model_path: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"Loading tokenizer from {model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model from {model_path}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_adapter:
        if not os.path.exists(ADAPTER_PATH):
            print(f"ERROR: adapter not found at {ADAPTER_PATH}", file=sys.stderr)
            sys.exit(1)
        from peft import PeftModel
        print(f"Loading adapter from {ADAPTER_PATH}...", flush=True)
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, prompts: list[str]) -> list[str]:
    import torch

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    responses = []
    for out in outputs:
        text = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        responses.append(text)
    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", action="store_true", help="Use adapter from data/adapter_tmp/")
    parser.add_argument("--model-path", default=_DEFAULT_MODEL_PATH, help="Path to model (default: Qwen3.5-0.8B)")
    args = parser.parse_args()
    MODEL_PATH = args.model_path if os.path.isabs(args.model_path) else os.path.join(ROOT, args.model_path)

    # Import here so prints above appear before the heavy evaluate.py module loads
    sys.path.insert(0, ROOT)
    from evaluate import verify_instruction

    def passes_all(item: dict, response: str) -> bool:
        instruction_ids = item.get("instruction_id_list", [])
        kwargss = item.get("kwargs", [{}] * len(instruction_ids))
        return all(
            verify_instruction(iid, kw if kw else {}, response)
            for iid, kw in zip(instruction_ids, kwargss)
        )

    prompts = []
    with open(IFEVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))

    print(f"Loaded {len(prompts)} IFEval prompts", flush=True)
    print(f"Generating {N_SAMPLES} responses per prompt ({len(prompts) * N_SAMPLES} total)...", flush=True)

    model, tokenizer = load_model(use_adapter=args.adapter, model_path=MODEL_PATH)

    kept = []
    # passed_responses[i] accumulates passing responses for prompts[i]
    passed_responses = [[] for _ in range(len(prompts))]

    # N_SAMPLES rounds, each round batches all 541 prompts in groups of BATCH_SIZE
    # This mirrors evaluate.py: batch different prompts together for GPU efficiency
    for sample_round in range(N_SAMPLES):
        round_kept = 0
        n_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx, batch_start in enumerate(range(0, len(prompts), BATCH_SIZE)):
            batch = prompts[batch_start : batch_start + BATCH_SIZE]
            batch_texts = [item["prompt"] for item in batch]
            responses = generate_batch(model, tokenizer, batch_texts)
            for i, (item, resp) in enumerate(zip(batch, responses)):
                if passes_all(item, resp):
                    passed_responses[batch_start + i].append(resp)
                    round_kept += 1
            done = batch_start + len(batch)
            print(
                f"  [round {sample_round+1}/{N_SAMPLES} | batch {batch_idx+1}/{n_batches} | {done}/{len(prompts)} prompts] "
                f"passing so far this round: {round_kept}",
                flush=True,
            )

        print(
            f"  === round {sample_round+1} done: {round_kept} passing | "
            f"prompts with >=1 pass: {sum(1 for p in passed_responses if p)} ===",
            flush=True,
        )

    for idx, (item, responses) in enumerate(zip(prompts, passed_responses)):
        for resp in responses:
            kept.append({"instruction": item["prompt"], "output": resp})

    prompts_with_pass = sum(1 for p in passed_responses if p)
    print(f"\nDone. {len(kept)} passing (prompt, response) pairs from {prompts_with_pass}/{len(prompts)} prompts.")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved -> {OUT_PATH}")
    print("Set TRAIN_DATA in finetune.py to: data/train_ifeval_rejection.jsonl")


if __name__ == "__main__":
    main()
