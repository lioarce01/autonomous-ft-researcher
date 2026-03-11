# Plan: IFEval Fine-Tuning Researcher — Full Project Design

## Context

Build a second autonomous research loop, parallel in architecture to `autonomous-ml-trainer`,
focused on **fine-tuning Qwen3.5-2B on IFEval (Instruction Following Evaluation)**.
The agent maximizes **IFEval prompt-level accuracy** by systematically exploring LoRA configs,
training data strategies, and prompt formats. The baseline score of 61.2 (non-thinking mode)
vs Qwen3-4B's 83.4 shows a clear 22-point gap — real headroom to exploit.

The agent (Claude Code) reads `PROGRAM.md` as its system prompt, edits `finetune.py`,
runs experiments within a 20-minute budget, logs results to SQLite, and iterates.
Infrastructure mirrors `autonomous-ml-trainer` exactly: same db.py/log_result.py/context_gen.py
pipeline, same NOTES.md + MEMORY_LOG.md pattern, same XML-tag PROGRAM.md structure.

For references, check github repo: https://github.com/lioarce01/autonomous-ml-researcher

---

## Why IFEval Over GSM8K

| | GSM8K | IFEval |
|---|---|---|
| Qwen3.5-2B baseline | ~85-90% (saturated, not reported) | **61.2%** (explicitly reported) |
| Gap to Qwen3-4B | unknown / minimal | **−22 pts** |
| Training data availability | abundant | abundant (UltraFeedback, Alpaca, etc.) |
| Fine-tune signal clarity | good | excellent — binary verifiable constraints |
| Research interest | low (solved) | high (active area) |

---

## What Is IFEval

IFEval (Zhou et al. 2023, arxiv:2311.07911) consists of **541 prompts**, each with one or more
**verifiable instruction constraints** such as:
- "Write more than 500 words"
- "Do not use the word 'example'"
- "Your response must contain exactly 3 bullet points"
- "Start your response with the letter Q"
- "Write in JSON format"

Evaluation is fully automatic — no LLM judge needed. Two metrics:
- **prompt_level_strict_acc**: ALL instructions in a prompt must be satisfied (primary metric)
- **instruction_level_strict_acc**: fraction of individual instructions satisfied

We use **prompt_level_strict_acc** as the primary metric (harder, cleaner signal).

---

## Project Structure

```
ifeval-finetune-researcher/
├── PROGRAM.md           ← Agent system prompt (XML tags, mirrors autonomous-ml-trainer)
├── CONTEXT.md           ← Auto-generated after each experiment
├── NOTES.md             ← Agent's persistent research notebook (writable)
├── MEMORY_LOG.md        ← Append-only VRAM anomaly log (writable)
├── finetune.py          ← QLoRA fine-tuning script — agent edits this each experiment
├── evaluate.py          ← Runs IFEval on adapter output, prints accuracy
├── prepare.py           ← Downloads Qwen3.5-2B + IFEval + training data (READ ONLY)
├── log_result.py        ← CLI: --name --accuracy --notes --hypothesis → DB + CONTEXT.md
├── context_gen.py       ← Reads DB, writes CONTEXT.md
├── db.py                ← sqlite3 stdlib wrapper
├── dashboard.py         ← Streamlit live dashboard
├── requirements.txt
└── data/
    ├── experiments.db
    ├── ifeval_prompts.jsonl       ← 541 IFEval prompts + instruction verifiers
    ├── train_alpaca.jsonl         ← 52k Alpaca instruction-following examples
    ├── train_ultrafeedback.jsonl  ← 10k UltraFeedback high-quality subset
    └── models/
        └── Qwen3.5-2B/            ← cached base weights (~4 GB)
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Base model | `Qwen/Qwen3.5-2B` | Model from ss.md; 61.2 IFEval baseline — clear headroom |
| Fine-tuning method | QLoRA (4-bit NF4 + LoRA) | Fits RTX 5070; standard baseline |
| Metric | IFEval prompt_level_strict_acc (0.0–1.0) | Binary, auto-verifiable, no LLM judge |
| Training mode | Non-thinking (no `<think>` tags) | Baseline 61.2 is non-thinking; consistent comparison |
| Budget | `BUDGET_SECONDS = 1200` (20 min) | ~15 min training + ~4 min eval on 541 prompts |
| Output contract | `print(f"accuracy: {accuracy:.4f}")` | Mirrors `val_bpb` pattern |

---

## Metric & Output Contract

```python
# evaluate.py prints — agent reads with: grep "accuracy:" run.log
print(f"accuracy: {accuracy:.4f}")   # e.g. accuracy: 0.6474
```

IFEval evaluation approach:
- Generate responses for all 541 prompts with greedy decoding (temperature=0)
- Run the official IFEval verifiers (pure Python, no external calls)
- Verifiers check: word count, keyword presence/absence, format (JSON, bullets, etc.), case, length
- Score = prompts where ALL constraints satisfied / 541

---

## Important Architecture Note: Qwen3.5-2B Is a Hybrid Model

Qwen3.5-2B uses a non-standard hybrid architecture:
```
24 layers = 6 × (
    3 × Gated DeltaNet → FFN    ← linear attention variant
    1 × Gated Attention → FFN   ← standard attention
)
```
Only **6 out of 24 layers** are standard attention (q_proj, k_proj, v_proj, o_proj).
The other 18 layers are Gated DeltaNet with different projections.

**Implication for LoRA targeting:**
- Baseline: target only Gated Attention layers (q_proj, v_proj in the 1/4 attention layers)
- Tier 1: also target k_proj, o_proj in Gated Attention layers
- Tier 2: also target DeltaNet layers (different param names — inspect with WebSearch/model card)
- Always WebSearch "Qwen3.5 LoRA target modules" before expanding beyond attention layers

---

## Baseline LoRA Config (finetune.py defaults)

```python
# ── Model ─────────────────────────────────────────
MODEL_NAME       = "data/models/Qwen3.5-2B"
BUDGET_SECONDS   = 1200   # 20-min wall-clock; NEVER CHANGE

# ── QLoRA quantization ────────────────────────────
LOAD_IN_4BIT     = True
BNB_4BIT_COMPUTE = "bfloat16"
BNB_4BIT_QUANT   = "nf4"

# ── LoRA adapter ──────────────────────────────────
LORA_RANK        = 8
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]   # attention layers only (baseline)

# ── Training ──────────────────────────────────────
TRAIN_DATA       = "data/train_alpaca.jsonl"  # start with clean Alpaca
TRAIN_SAMPLES    = 10_000                     # subset for speed
LEARNING_RATE    = 2e-4
BATCH_SIZE       = 4
GRAD_ACCUM       = 8                          # effective batch = 32
MAX_EPOCHS       = 2
WARMUP_RATIO     = 0.03
LR_SCHEDULER     = "cosine"
MAX_SEQ_LEN      = 512
BF16             = True
THINKING_MODE    = False                      # disable <think> tokens during generation
```

---

## Prompt Format (Baseline)

```python
SYSTEM_PROMPT = "You are a helpful assistant. Follow all instructions precisely."

def format_train(instruction, output):
    return f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>"""
```

For evaluation (IFEval), no system prompt modification — use prompts exactly as provided.

---

## Exploration Guide (5 Tiers)

### Tier 1 — High Impact

| Technique | Experiment name | Notes |
|---|---|---|
| Expand LoRA target modules | `target_full_attn` | Add k_proj + o_proj to all Gated Attention layers |
| Higher LoRA rank | `lora_r16`, `lora_r32` | More adapter capacity |
| Switch training data | `data_ultrafeedback` | UltraFeedback 10k vs Alpaca 10k |
| Constraint-aware data | `data_constraint_filtered` | Filter Alpaca for examples with explicit format/length constraints |

### Tier 2 — Medium Impact

| Technique | Experiment name | Notes |
|---|---|---|
| Learning rate | `lr_1e4`, `lr_5e4` | 1e-4 (more conservative) or 5e-4 (aggressive) |
| More training samples | `samples_30k`, `samples_50k` | Scale from 10k to 30k/50k |
| More epochs | `epochs_3`, `epochs_5` | Small dataset may benefit |
| System prompt with constraints | `sysprompt_strict` | "Follow ALL constraints exactly. Check word count, format, and forbidden words before responding." |
| LR scheduler | `sched_linear`, `sched_constant` | Alternative to cosine |
| Gradient accumulation | `grad8_batch8` | Larger effective batch |

### Tier 3 — Lower Impact

| Technique | Experiment name | Notes |
|---|---|---|
| LoRA alpha | `alpha_8` (ratio=1.0) or `alpha_32` (ratio=4.0) | Scaling factor tuning |
| LoRA dropout | `lora_drop_0` | No dropout may help on clean data |
| 8-bit quantization | `qlora_8bit` | More capacity, more VRAM |
| Max seq length | `seqlen_256`, `seqlen_768` | Shorter = faster; longer = full constraint check space |
| Warmup ratio | `warmup_05`, `warmup_10` | |

### Tier 4 — Advanced

| Technique | Flags | Notes |
|---|---|---|
| DoRA | AMBITIOUS | `use_dora=True` in PEFT; weight-decomposed; often +1-3 pts |
| RSLoRA | EASY | `use_rslora=True`; scale by 1/sqrt(r); try at rank=32 |
| DeltaNet layer targeting | RESEARCH | Target linear attention layers; WebSearch param names first |
| Data mixing | MEDIUM | Combine Alpaca + UltraFeedback + constraint-synthetic data |

### Tier 5 — Novel

- Synthetic constraint data: generate 1k examples where each output follows an explicit constraint (GPT-4/Claude generated); fine-tune on these directly
- Thinking mode fine-tuning: train with `THINKING_MODE=True` and compare vs non-thinking
- Format-specific tuning: oversample JSON/bullet/numbered-list examples from training data
- Rejection sampling: generate 5 responses per IFEval train-style prompt, keep only constraint-satisfying ones, fine-tune on those

---

## The Experiment Loop (for PROGRAM.md)

Steps map directly to `autonomous-ml-trainer`, with these changes:
- Editable file: `finetune.py` (not `train.py`)
- Step 6: `uv run python finetune.py > run.log 2>&1` (trains adapter, saves to `data/adapter_tmp/`)
- Step 7: `grep "accuracy:" run.log`
- Step 8: `log_result.py --name NAME --accuracy X.XXXX --notes "..." --hypothesis "..."`
- Metric direction: **higher is better** (db keeps if accuracy > previous best)
- Budget: 20 min total

---

## Files to Create — Implementation Details

### `db.py`
Same structure as autonomous-ml-trainer, but:
- Column: `accuracy REAL` (not `val_bpb`)
- `kept=1` when `accuracy > MAX(accuracy) WHERE kept=1`

### `log_result.py`
CLI: `--name`, `--accuracy`, `--notes`, `--hypothesis`
Prints: kept YES/NO, best accuracy, suggested git commit

### `context_gen.py`
Generates CONTEXT.md sections:
- Current best (name, accuracy, notes, delta from baseline 0.612)
- Leaderboard table (rank, name, accuracy, delta, notes, timestamp, kept)
- Recent failures (last 3 not kept)
- Unexplored techniques (check experiment names against technique list)
- Memory anomalies (MEMORY_LOG.md)
- NOTES.md verbatim

### `prepare.py`
```python
# 1. Download Qwen3.5-2B weights → data/models/Qwen3.5-2B/
# 2. Download IFEval prompts → data/ifeval_prompts.jsonl (541 prompts + verifier metadata)
#    Source: HuggingFace google/ifeval or direct from github.com/google-deepmind/...
# 3. Download Alpaca dataset → data/train_alpaca.jsonl (52k examples)
#    Source: tatsu-lab/alpaca on HuggingFace
# 4. Download UltraFeedback → sample 10k high-scored examples → data/train_ultrafeedback.jsonl
#    Source: openbmb/UltraFeedback on HuggingFace, filter score >= 4.5
```

### `finetune.py`
```
- Load Qwen3.5-2B in 4-bit QLoRA
- Apply LoRA via peft.get_peft_model()
- Load training data (TRAIN_DATA path), format with ChatML template
- Train until BUDGET_SECONDS * 0.75 (leaves time for eval)
- Save adapter to data/adapter_tmp/
- Run evaluate.py inline or via subprocess
- Print: accuracy: X.XXXX
```

### `evaluate.py`
```
- Load Qwen3.5-2B in 4-bit + adapter from data/adapter_tmp/
- For each of 541 IFEval prompts:
    - Generate response with greedy decoding, THINKING_MODE=False
    - Run verifiers: check each constraint in the prompt
- prompt_level_strict_acc = prompts where ALL constraints pass / 541
- Print: accuracy: X.XXXX
- Print: instruction_level_acc: X.XXXX (secondary, informational)
```

IFEval verifier categories (pure Python, no external calls):
- `keyword`: must/must not contain word X
- `length_constraint`: word count, sentence count, paragraph count
- `detectable_format`: JSON, markdown headers, bullet points, title case
- `startend`: starts/ends with specific string
- `language`: response language
- `combination`: multiple constraints AND-ed

### `PROGRAM.md`
XML-tag structure identical to autonomous-ml-trainer. Key differences:
```
Role: maximize IFEval prompt_level_strict_acc on Qwen3.5-2B
Metric: accuracy (higher = better, 0.0-1.0)
Baseline: 0.612 (Qwen3.5-2B non-thinking, from model card)
Budget: 20 min
Editable file: finetune.py
Output contract: print(f"accuracy: {accuracy:.4f}")
Naming: lora_r16, data_ultrafeedback, target_full_attn, lr_5e4, sysprompt_strict, etc.
```

### `requirements.txt`
```
--extra-index-url https://download.pytorch.org/whl/cu128

torch>=2.7.0
transformers>=4.45.0
peft>=0.13.0
bitsandbytes>=0.43.0
accelerate>=0.34.0
datasets>=2.14.0
trl>=0.11.0
numpy>=1.24.0
streamlit>=1.32.0
plotly>=5.18.0
pandas>=2.0.0
```

---

## Day-0 Setup Checklist

```bash
# 1. Create and populate environment
uv venv .venv && uv pip install -r requirements.txt

# 2. Download model + data (~4 GB model + small datasets)
uv run python prepare.py

# 3. Run baseline (no adapter — raw Qwen3.5-2B)
uv run python evaluate.py --no-adapter > run.log 2>&1
grep "accuracy:" run.log
# Expected: ~0.612 (matches model card)

# 4. Log baseline
uv run python log_result.py --name "baseline_raw" --accuracy 0.6120 \
  --notes "Raw Qwen3.5-2B, no fine-tuning. Model card reports 61.2 non-thinking IFEval."

# 5. Run first fine-tuning experiment
uv run python finetune.py > run.log 2>&1
grep "accuracy:" run.log

# 6. Log result + commit
uv run python log_result.py --name "baseline_qlora" --accuracy X.XXXX \
  --notes "Baseline QLoRA: rank=8, alpha=16, target=q+v, Alpaca 10k, lr=2e-4, ep=2. CoT off."
git commit ... (use suggested command)

# 7. Begin research loop
```

---

## Curated Papers for PROGRAM.md

| Technique | Paper | Key finding |
|---|---|---|
| IFEval benchmark | Zhou et al. 2023 arxiv:2311.07911 | 541 prompts with verifiable constraints; prompt-level strict acc most reliable metric |
| LoRA | Hu et al. 2022 arxiv:2106.09685 | Low-rank adaptation; r=8 baseline; A+B matrices only; freeze base model |
| QLoRA | Dettmers et al. 2023 arxiv:2305.14314 | 4-bit NF4 quantization + LoRA; matches full fine-tune quality; standard SFT baseline |
| DoRA | Liu et al. 2024 arxiv:2402.09353 | Weight-decomposed LoRA; separate magnitude/direction; often outperforms LoRA at same rank |
| RSLoRA | Kalajdzievski 2024 arxiv:2312.03732 | Scale by 1/sqrt(r); better at high ranks; `use_rslora=True` in PEFT — one flag change |
| LoRA+ | Hayou et al. 2024 arxiv:2402.12354 | Different LR for A (lower) and B (higher) matrices; simple, often improves convergence |
| UltraFeedback | Cui et al. 2023 arxiv:2310.01377 | 256k diverse instruction-following annotations; high-quality subset (score ≥4.5) best for SFT |
| Alpaca | Taori et al. 2023 alpaca project | 52k GPT-4-generated instruction-output pairs; clean baseline SFT data |
| Self-Instruct | Wang et al. 2022 arxiv:2212.10560 | Bootstrap instruction data from LLM itself; useful for generating constraint-rich examples |
| Qwen3.5-2B | Qwen team 2026 | Hybrid Gated DeltaNet + sparse attention; 2B params; 61.2 IFEval (non-thinking); 201 languages |

---

## Files to Create

| File | Status |
|---|---|
| `ifeval-finetune-researcher/PROGRAM.md` | CREATE — XML-tag system prompt |
| `ifeval-finetune-researcher/finetune.py` | CREATE — QLoRA training script |
| `ifeval-finetune-researcher/evaluate.py` | CREATE — IFEval verifier + scoring |
| `ifeval-finetune-researcher/prepare.py` | CREATE — data + model download |
| `ifeval-finetune-researcher/log_result.py` | CREATE — logging CLI |
| `ifeval-finetune-researcher/context_gen.py` | CREATE — CONTEXT.md generator |
| `ifeval-finetune-researcher/db.py` | CREATE — SQLite wrapper |
| `ifeval-finetune-researcher/dashboard.py` | CREATE — Streamlit dashboard |
| `ifeval-finetune-researcher/requirements.txt` | CREATE |
| `ifeval-finetune-researcher/NOTES.md` | CREATE — empty template |
| `ifeval-finetune-researcher/MEMORY_LOG.md` | CREATE — empty with column headers |

No files in `autonomous-ml-trainer/` are modified.

---

## Verification

1. `uv run python prepare.py` — model + datasets in data/
2. `uv run python evaluate.py --no-adapter` — prints ~0.612 (matches model card baseline)
3. `uv run python finetune.py > run.log 2>&1` — completes in ~20 min, `grep "accuracy:" run.log` returns value
4. `uv run python log_result.py --name test --accuracy 0.65 --notes "test"` — CONTEXT.md generated
5. `uv run streamlit run dashboard.py` — live chart shows
6. PROGRAM.md spot-check: all steps, output contract, all 5 tier labels present
