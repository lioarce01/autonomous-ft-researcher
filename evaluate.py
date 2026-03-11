"""
IFEval evaluator for Qwen3.5-2B (with or without LoRA adapter).

Usage:
    uv run python evaluate.py                # with adapter from data/adapter_tmp/
    uv run python evaluate.py --no-adapter   # raw base model (baseline check)

Output (last two lines, for grep):
    accuracy: X.XXXX
    instruction_level_acc: X.XXXX
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(ROOT, "data", "models", "Qwen3.5-2B")
ADAPTER_PATH  = os.path.join(ROOT, "data", "adapter_tmp")
IFEVAL_PATH   = os.path.join(ROOT, "data", "ifeval_prompts.jsonl")

THINKING_MODE  = False  # never use <think> tokens for consistent comparison
EVAL_BATCH_SIZE = 16    # prompts generated in parallel; lower if OOM


# IFEval Verifiers

def verify_instruction(instruction_id: str, kwargs: dict, response: str) -> bool:
    """Return True if the response satisfies the given IFEval instruction."""
    resp = response.strip()

    # keyword
    if instruction_id == "keywords:existence":
        keywords = kwargs.get("keywords", [])
        return all(kw.lower() in resp.lower() for kw in keywords)

    if instruction_id == "keywords:frequency":
        keyword = kwargs.get("keyword", "")
        freq = kwargs.get("frequency", 0)
        relation = kwargs.get("relation", "at least")
        count = resp.lower().count(keyword.lower())
        if relation == "at least":
            return count >= freq
        if relation == "at most":
            return count <= freq
        return count == freq

    if instruction_id == "keywords:forbidden_words":
        forbidden = kwargs.get("forbidden_words", [])
        return all(fw.lower() not in resp.lower() for fw in forbidden)

    if instruction_id == "keywords:letter_frequency":
        letter = kwargs.get("letter", "").lower()
        freq = kwargs.get("let_frequency", 0)
        relation = kwargs.get("let_relation", "at least")
        count = resp.lower().count(letter)
        if relation == "at least":
            return count >= freq
        if relation == "at most":
            return count <= freq
        return count == freq

    # length_constraint
    if instruction_id == "length_constraint:number_sentences":
        sentences = [s.strip() for s in re.split(r'[.!?]+', resp) if s.strip()]
        num = len(sentences)
        relation = kwargs.get("relation", "at least")
        target = kwargs.get("num_sentences", 0)
        if relation == "at least":
            return num >= target
        if relation == "at most":
            return num <= target
        return num == target

    if instruction_id == "length_constraint:number_paragraphs":
        paragraphs = [p.strip() for p in resp.split("\n\n") if p.strip()]
        num = len(paragraphs)
        relation = kwargs.get("relation", "at least")
        target = kwargs.get("num_paragraphs", 0)
        if relation == "at least":
            return num >= target
        if relation == "at most":
            return num <= target
        return num == target

    if instruction_id == "length_constraint:number_words":
        words = resp.split()
        num = len(words)
        relation = kwargs.get("relation", "at least")
        target = kwargs.get("num_words", 0)
        if relation == "at least":
            return num >= target
        if relation == "at most":
            return num <= target
        return num == target

    if instruction_id == "length_constraint:nth_paragraph_first_word":
        n = kwargs.get("nth_paragraph", 1)
        first_word = kwargs.get("first_word", "")
        paragraphs = [p.strip() for p in resp.split("\n\n") if p.strip()]
        if n > len(paragraphs):
            return False
        para = paragraphs[n - 1]
        words = para.split()
        return bool(words) and words[0].lower() == first_word.lower()

    # detectable_format
    if instruction_id == "detectable_format:number_bullet_lists":
        bullet_lines = [l for l in resp.split("\n") if re.match(r'^\s*[-*]\s+', l)]
        num = len(bullet_lines)
        relation = kwargs.get("relation", "at least")
        target = kwargs.get("num_bullets", 0)
        if relation == "at least":
            return num >= target
        if relation == "at most":
            return num <= target
        return num == target

    if instruction_id == "detectable_format:constrained_response":
        options = kwargs.get("options", [])
        return resp.strip() in options

    if instruction_id == "detectable_format:number_highlighted_sections":
        highlighted = re.findall(r'\*[^*]+\*', resp)
        num = len(highlighted)
        relation = kwargs.get("relation", "at least")
        target = kwargs.get("num_highlights", 0)
        if relation == "at least":
            return num >= target
        if relation == "at most":
            return num <= target
        return num == target

    if instruction_id == "detectable_format:multiple_sections":
        headers = re.findall(r'^#{1,6}\s+\S', resp, re.MULTILINE)
        num = len(headers)
        relation = kwargs.get("relation", "at least")
        target = kwargs.get("num_sections", 0)
        if relation == "at least":
            return num >= target
        if relation == "at most":
            return num <= target
        return num == target

    if instruction_id == "detectable_format:json_format":
        try:
            match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', resp)
            if match:
                json.loads(match.group(1))
                return True
        except (json.JSONDecodeError, AttributeError):
            pass
        return False

    if instruction_id == "detectable_format:title":
        for line in resp.split("\n"):
            line = line.strip()
            if line and line == line.title():
                return True
        return False

    # startend
    if instruction_id == "startend:end_checker":
        end_phrase = kwargs.get("end_phrase", "")
        return resp.endswith(end_phrase)

    if instruction_id == "startend:punctuation":
        punct = kwargs.get("punctuation", "")
        return resp.endswith(punct)

    if instruction_id == "startend:quotation":
        return resp.startswith('"') and resp.endswith('"')

    # detectable_content
    if instruction_id == "detectable_content:number_placeholders":
        placeholders = re.findall(r'\[[^\]]+\]', resp)
        num = len(placeholders)
        relation = kwargs.get("relation", "at least")
        target = kwargs.get("num_placeholders", 0)
        if relation == "at least":
            return num >= target
        if relation == "at most":
            return num <= target
        return num == target

    if instruction_id == "detectable_content:postscript":
        postscript_marker = kwargs.get("postscript_marker", "P.S.")
        return postscript_marker in resp

    # language
    if instruction_id == "language:response_language":
        try:
            import langdetect
            lang = langdetect.detect(resp)
            return lang == kwargs.get("language", "en")
        except Exception:
            required = kwargs.get("language", "en")
            if required == "en":
                ascii_ratio = sum(1 for c in resp if ord(c) < 128) / max(len(resp), 1)
                return ascii_ratio > 0.9
            return True

    # combination
    if instruction_id == "combination:two_responses":
        return "****" in resp

    if instruction_id == "combination:repeat_prompt":
        prompt_text = kwargs.get("prompt_to_repeat", "")
        return prompt_text.lower() in resp.lower() if prompt_text else False

    # change_case
    if instruction_id == "change_case:capital_word_frequency":
        capital_words = [w for w in resp.split() if w.isupper() and len(w) > 1]
        num = len(capital_words)
        relation = kwargs.get("capital_relation", "at least")
        target = kwargs.get("capital_frequency", 0)
        if relation == "at least":
            return num >= target
        if relation == "at most":
            return num <= target
        return num == target

    if instruction_id == "change_case:english_capital":
        return resp == resp.upper()

    if instruction_id == "change_case:english_lowercase":
        return resp == resp.lower()

    # punctuation
    if instruction_id == "punctuation:no_comma":
        return "," not in resp

    # Unknown instruction - give benefit of the doubt
    return True


# Model Loading

def load_model(use_adapter: bool):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"Loading tokenizer from {MODEL_PATH}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # Left-pad so all prompts in a batch align on the right (generation side)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model from {MODEL_PATH}...", flush=True)
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
        print(f"Loading LoRA adapter from {ADAPTER_PATH}...", flush=True)
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()
    return model, tokenizer


# Batch Generation

def generate_batch(model, tokenizer, prompt_texts: list[str], max_new_tokens: int = 512) -> list[str]:
    """Generate responses for a batch of prompts in one forward pass."""
    import torch

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompt_texts
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
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    responses = []
    for out in outputs:
        new_tokens = out[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(text)

    return responses


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-adapter", action="store_true", help="Evaluate raw base model")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit to N prompts (debug)")
    args = parser.parse_args()

    if not os.path.exists(IFEVAL_PATH):
        print(f"ERROR: IFEval prompts not found at {IFEVAL_PATH}", file=sys.stderr)
        print("Run: uv run python prepare.py", file=sys.stderr)
        sys.exit(1)

    prompts = []
    with open(IFEVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    if args.max_samples:
        prompts = prompts[: args.max_samples]

    print(f"Loaded {len(prompts)} IFEval prompts", flush=True)

    model, tokenizer = load_model(use_adapter=not args.no_adapter)

    prompt_correct = 0
    instruction_correct = 0
    instruction_total = 0

    for batch_start in range(0, len(prompts), EVAL_BATCH_SIZE):
        batch = prompts[batch_start : batch_start + EVAL_BATCH_SIZE]
        batch_prompts = [item["prompt"] for item in batch]

        responses = generate_batch(model, tokenizer, batch_prompts)

        for item, response in zip(batch, responses):
            instruction_ids = item.get("instruction_id_list", [])
            kwargss = item.get("kwargs", [{}] * len(instruction_ids))

            instr_results = [
                verify_instruction(instr_id, kw if kw else {}, response)
                for instr_id, kw in zip(instruction_ids, kwargss)
            ]

            if all(instr_results):
                prompt_correct += 1
            instruction_correct += sum(instr_results)
            instruction_total += len(instr_results)

        done = min(batch_start + EVAL_BATCH_SIZE, len(prompts))
        print(
            f"  [{done}/{len(prompts)}] prompt_acc={prompt_correct/done:.4f}",
            flush=True,
        )

    prompt_acc = prompt_correct / len(prompts)
    instr_acc = instruction_correct / max(instruction_total, 1)

    print(f"\nResults on {len(prompts)} prompts:")
    print(f"accuracy: {prompt_acc:.4f}")
    print(f"instruction_level_acc: {instr_acc:.4f}")


if __name__ == "__main__":
    main()
