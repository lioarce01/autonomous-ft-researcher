"""
Rejection sampling on IFEval prompts.

For each of the 541 IFEval prompts, generate N_SAMPLES responses with temperature
sampling, run the verifiers, and keep only responses where ALL constraints pass.

This gives the highest-quality training signal: every kept example is a (prompt, response)
pair that our evaluator would score as correct.

Output: data/train_ifeval_rejection.jsonl
Format: {"instruction": "<ifeval prompt>", "output": "<passing response>"}

Usage:
    uv run python prepare_rejection.py               # use base model
    uv run python prepare_rejection.py --adapter     # use current adapter in data/adapter_tmp/
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(ROOT, "data", "models", "Qwen3.5-2B")
ADAPTER_PATH = os.path.join(ROOT, "data", "adapter_tmp")
IFEVAL_PATH  = os.path.join(ROOT, "data", "ifeval_prompts.jsonl")
OUT_PATH     = os.path.join(ROOT, "data", "train_ifeval_rejection.jsonl")

N_SAMPLES       = 8     # responses generated per prompt
TEMPERATURE     = 0.8   # sampling temperature for diversity
MAX_NEW_TOKENS  = 700   # slightly more headroom than eval for word-count constraints
BATCH_SIZE      = 4     # prompts per batch during generation


# Import verifiers from evaluate.py
sys.path.insert(0, ROOT)
from evaluate import verify_instruction


def load_model(use_adapter: bool):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"Loading tokenizer from {MODEL_PATH}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
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


def passes_all(item: dict, response: str) -> bool:
    instruction_ids = item.get("instruction_id_list", [])
    kwargss = item.get("kwargs", [{}] * len(instruction_ids))
    return all(
        verify_instruction(iid, kw if kw else {}, response)
        for iid, kw in zip(instruction_ids, kwargss)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", action="store_true", help="Use adapter from data/adapter_tmp/")
    args = parser.parse_args()

    prompts = []
    with open(IFEVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))

    print(f"Loaded {len(prompts)} IFEval prompts", flush=True)
    print(f"Generating {N_SAMPLES} responses per prompt ({len(prompts) * N_SAMPLES} total)...", flush=True)

    model, tokenizer = load_model(use_adapter=args.adapter)

    kept = []
    prompt_pass_count = 0

    for idx, item in enumerate(prompts):
        prompt_text = item["prompt"]
        passed_responses = []

        # Generate N_SAMPLES in mini-batches
        for batch_start in range(0, N_SAMPLES, BATCH_SIZE):
            batch_size = min(BATCH_SIZE, N_SAMPLES - batch_start)
            responses = generate_batch(model, tokenizer, [prompt_text] * batch_size)
            for resp in responses:
                if passes_all(item, resp):
                    passed_responses.append(resp)

        if passed_responses:
            prompt_pass_count += 1
            for resp in passed_responses:
                kept.append({"instruction": prompt_text, "output": resp})

        if (idx + 1) % 50 == 0 or idx == len(prompts) - 1:
            print(
                f"  [{idx+1}/{len(prompts)}] prompts with passing responses: {prompt_pass_count} | "
                f"total kept: {len(kept)}",
                flush=True,
            )

    print(f"\nDone. {len(kept)} passing (prompt, response) pairs from {prompt_pass_count}/{len(prompts)} prompts.")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved -> {OUT_PATH}")
    print("Set TRAIN_DATA in finetune.py to: data/train_ifeval_rejection.jsonl")


if __name__ == "__main__":
    main()
