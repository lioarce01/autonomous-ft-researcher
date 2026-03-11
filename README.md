# IFEval Fine-Tuning Researcher

Autonomous research loop that maximizes **IFEval prompt_level_strict_acc** on Qwen3.5-2B
by systematically exploring QLoRA configs, training data strategies, and prompt formats.

| | |
|---|---|
| **Base model** | Qwen3.5-2B (hybrid Gated DeltaNet + sparse attention) |
| **Baseline** | 0.612 (non-thinking mode, from model card) |
| **Target** | 0.834 (Qwen3-4B IFEval score) |
| **Method** | QLoRA (4-bit NF4 + LoRA) |
| **Budget** | 20 min per experiment |

---

## What is IFEval

IFEval ([Zhou et al. 2023](https://arxiv.org/abs/2311.07911)) is a benchmark of **541 prompts**, each with one or more
verifiable constraints: word count limits, forbidden words, required formats (JSON, bullets),
start/end strings, letter frequency, and more. Evaluation is fully automatic — no LLM judge needed.

Primary metric: **prompt_level_strict_acc** — fraction of prompts where ALL constraints are satisfied.

---

## Project Structure

```
.
├── PROGRAM.md           ← Agent system prompt (read this to run the loop)
├── CONTEXT.md           ← Auto-generated after each experiment (leaderboard, best, unexplored)
├── NOTES.md             ← Agent's persistent research notebook
├── MEMORY_LOG.md        ← Append-only VRAM anomaly log
├── finetune.py          ← QLoRA training script — agent edits this each experiment
├── evaluate.py          ← IFEval verifier + scoring (541 prompts, greedy decode)
├── prepare.py           ← Downloads IFEval prompts + training datasets (run once)
├── log_result.py        ← CLI: logs result to DB, regenerates CONTEXT.md
├── context_gen.py       ← Reads DB, writes CONTEXT.md
├── db.py                ← SQLite wrapper
├── dashboard.py         ← Streamlit live dashboard
├── requirements.txt
└── data/
    ├── experiments.db              ← experiment log
    ├── ifeval_prompts.jsonl        ← 541 IFEval prompts + verifier metadata
    ├── train_alpaca.jsonl          ← 52k Alpaca instruction-following examples
    ├── train_ultrafeedback.jsonl   ← 10k UltraFeedback high-quality subset (score ≥ 4.5)
    ├── adapter_tmp/                ← LoRA adapter output (current experiment)
    └── models/
        └── Qwen3.5-2B/            ← base model weights (~4 GB)
```

---

## Setup

```bash
# 1. Install dependencies
uv venv .venv && uv pip install -r requirements.txt

# 2. Download datasets (model weights already in data/models/Qwen3.5-2B/)
uv run python prepare.py

# 3. Verify baseline (~0.612)
uv run python evaluate.py --no-adapter > run.log 2>&1
grep "accuracy:" run.log

# 4. Log baseline
uv run python log_result.py --name baseline_raw --accuracy 0.6120 \
  --notes "Raw Qwen3.5-2B, no fine-tuning. Model card reports 61.2 non-thinking IFEval."
```

---

## Experiment Loop

```bash
# Edit the CONFIG section in finetune.py, then:
uv run python finetune.py > run.log 2>&1
grep "accuracy:" run.log

uv run python log_result.py --name <name> --accuracy <value> \
  --notes "what changed + observed" --hypothesis "why expected to work"

# If kept (new best), commit:
git add -A && git commit -m "exp: <name> acc=<value>"
```

See `PROGRAM.md` for the full agent loop, all 5 exploration tiers, naming conventions, and decision rules.

---

## Dashboard

```bash
uv run streamlit run dashboard.py
```

Live chart of all experiments vs baseline (0.612) and target (0.834).

---

## Architecture Note

Qwen3.5-2B is a **hybrid model** — only 6 of 24 layers are standard attention:

```
24 layers = 6 × (3 × Gated DeltaNet → FFN + 1 × Gated Attention → FFN)
```

The baseline LoRA targets only `q_proj` and `v_proj` in those 6 attention layers.
Expanding to DeltaNet layers requires researching the correct parameter names first.

---

## Exploration Tiers

| Tier | Examples |
|------|---------|
| 1 — High impact | `target_full_attn`, `lora_r16`, `lora_r32`, `data_ultrafeedback` |
| 2 — Medium | `lr_1e4`, `samples_30k`, `epochs_3`, `sysprompt_strict` |
| 3 — Lower | `alpha_32`, `lora_drop_0`, `seqlen_768` |
| 4 — Advanced | `dora`, `rslora`, `deltanet_layers`, `data_mixed` |
| 5 — Novel | synthetic constraint data, rejection sampling, format oversampling |
