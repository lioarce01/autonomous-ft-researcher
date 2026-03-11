"""
QLoRA fine-tuning script for Qwen3.5-2B on IFEval-style instruction data.

The agent edits the CONFIG section below, then runs:
    uv run python finetune.py > run.log 2>&1
and reads the result with:
    grep "accuracy:" run.log
"""
import os
import json
import time
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# CONFIG - agent edits this section each experiment
# ==============================================================================

# Model
MODEL_NAME       = os.path.join(ROOT, "data", "models", "Qwen3.5-2B")
BUDGET_SECONDS   = 1200   # 20-min wall-clock; NEVER CHANGE
ADAPTER_OUT      = os.path.join(ROOT, "data", "adapter_tmp")

# QLoRA quantization
LOAD_IN_4BIT     = True
BNB_4BIT_COMPUTE = "bfloat16"
BNB_4BIT_QUANT   = "nf4"

# LoRA adapter
LORA_RANK        = 8
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]   # all attention projections

# Training
TRAIN_DATA       = os.path.join(ROOT, "data", "train_alpaca.jsonl")
TRAIN_SAMPLES    = 5_000
LEARNING_RATE    = 2e-4
BATCH_SIZE       = 4
GRAD_ACCUM       = 8    # effective batch = BATCH_SIZE * GRAD_ACCUM
MAX_EPOCHS       = 2
WARMUP_RATIO     = 0.03
LR_SCHEDULER     = "cosine"
MAX_SEQ_LEN      = 256
BF16             = True
THINKING_MODE    = False   # disable <think> tokens during evaluation

# Prompt
SYSTEM_PROMPT = "You are a helpful assistant. Follow all instructions precisely."

# ==============================================================================


def format_train(instruction: str, output: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{output}\n<|im_end|>"
    )


def load_training_data():
    if not os.path.exists(TRAIN_DATA):
        print(f"ERROR: training data not found at {TRAIN_DATA}", file=sys.stderr)
        print("Run: uv run python prepare.py", file=sys.stderr)
        sys.exit(1)

    examples = []
    with open(TRAIN_DATA, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            instruction = row.get("instruction", "")
            output = row.get("output", "")
            if instruction and output:
                examples.append(format_train(instruction, output))
            if len(examples) >= TRAIN_SAMPLES:
                break

    print(f"Loaded {len(examples)} training examples from {TRAIN_DATA}")
    return examples


def train(start_time: float):
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import Dataset
    from transformers import TrainerCallback

    class BudgetCallback(TrainerCallback):
        """Stop training when wall-clock budget is exhausted."""
        def __init__(self, deadline: float):
            self.deadline = deadline

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() >= self.deadline:
                print(f"\nBudget reached at step {state.global_step}. Stopping training.", flush=True)
                control.should_training_stop = True
            return control

    os.makedirs(ADAPTER_OUT, exist_ok=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import torch as th
    compute_dtype = th.bfloat16 if BNB_4BIT_COMPUTE == "bfloat16" else th.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=BNB_4BIT_QUANT,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model in 4-bit...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=compute_dtype,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_texts = load_training_data()

    class TextDataset(Dataset):
        def __init__(self, texts, tok, max_len):
            self.encodings = tok(
                texts,
                truncation=True,
                max_length=max_len,
                padding=False,
                return_tensors=None,
            )

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}

    dataset = TextDataset(raw_texts, tokenizer, MAX_SEQ_LEN)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    elapsed = time.time() - start_time
    # Use 75% of budget for training; reserve 25% for evaluation
    train_budget = BUDGET_SECONDS * 0.75 - elapsed
    if train_budget < 60:
        print("WARNING: less than 60s left for training, skipping straight to eval", flush=True)
        train_budget = 60

    deadline = time.time() + train_budget
    steps_per_epoch = len(dataset) // (BATCH_SIZE * GRAD_ACCUM)
    max_steps = steps_per_epoch * MAX_EPOCHS  # BudgetCallback will stop early by wall-clock
    print(f"Training budget: {train_budget:.0f}s, max steps (ceiling): {max_steps}", flush=True)

    training_args = TrainingArguments(
        output_dir=ADAPTER_OUT,
        max_steps=max_steps,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        bf16=BF16,
        fp16=False,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[BudgetCallback(deadline)],
    )

    print("Starting training...", flush=True)
    trainer.train()
    print("Training complete. Saving adapter...", flush=True)
    model.save_pretrained(ADAPTER_OUT)
    tokenizer.save_pretrained(ADAPTER_OUT)
    print(f"Adapter saved to {ADAPTER_OUT}", flush=True)

    del model
    del trainer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_evaluate():
    print("\nRunning evaluation...", flush=True)
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "evaluate.py")],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: evaluate.py exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    start = time.time()
    train(start)
    run_evaluate()
