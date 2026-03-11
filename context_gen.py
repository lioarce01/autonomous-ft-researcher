"""Generate CONTEXT.md from the experiments DB + NOTES.md + MEMORY_LOG.md."""
import os
from db import get_all_experiments, get_recent_not_kept

ROOT = os.path.dirname(os.path.abspath(__file__))
CONTEXT_PATH = os.path.join(ROOT, "CONTEXT.md")
NOTES_PATH = os.path.join(ROOT, "NOTES.md")
MEMORY_LOG_PATH = os.path.join(ROOT, "MEMORY_LOG.md")
BASELINE = 0.612

EXPLORED_TECHNIQUES = [
    "target_full_attn", "lora_r16", "lora_r32",
    "data_ultrafeedback", "data_constraint_filtered",
    "lr_1e4", "lr_5e4", "samples_30k", "samples_50k",
    "epochs_3", "epochs_5", "sysprompt_strict",
    "sched_linear", "sched_constant", "grad8_batch8",
    "alpha_8", "alpha_32", "lora_drop_0", "qlora_8bit",
    "seqlen_256", "seqlen_768", "warmup_05", "warmup_10",
    "dora", "rslora", "deltanet_layers", "data_mixed",
]


def _read_file(path: str) -> str:
    if not os.path.exists(path):
        return "(file not found)"
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def generate():
    experiments = get_all_experiments()
    recent_failures = get_recent_not_kept(3)

    kept = [e for e in experiments if e["kept"] == 1]
    best = kept[0] if kept else None

    lines = []

    # Section 1: Current Best
    lines.append("# CONTEXT - IFEval Fine-Tuning Researcher\n")
    lines.append("## Current Best\n")
    if best:
        delta = best["accuracy"] - BASELINE
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        lines.append(f"- **Name**: `{best['name']}`")
        lines.append(f"- **Accuracy**: {best['accuracy']:.4f}  (baseline {BASELINE:.3f}, delta {delta_str})")
        lines.append(f"- **Notes**: {best['notes']}")
        lines.append(f"- **Timestamp**: {best['timestamp']}")
    else:
        lines.append("- No experiment kept yet. Baseline: 0.612")
    lines.append("")

    # Section 2: Leaderboard
    lines.append("## Leaderboard\n")
    lines.append("| Rank | Name | Accuracy | Delta | Kept | Notes | Timestamp |")
    lines.append("|------|------|----------|-------|------|-------|-----------|")
    for rank, exp in enumerate(experiments, 1):
        delta = exp["accuracy"] - BASELINE
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        kept_str = "YES" if exp["kept"] == 1 else ""
        notes_short = (exp["notes"] or "")[:60].replace("|", "/")
        lines.append(
            f"| {rank} | `{exp['name']}` | {exp['accuracy']:.4f} | {delta_str} | {kept_str} | {notes_short} | {exp['timestamp']} |"
        )
    lines.append("")

    # Section 3: Recent Failures
    lines.append("## Recent Failures (not kept)\n")
    if recent_failures:
        for exp in recent_failures:
            lines.append(f"- `{exp['name']}` -> {exp['accuracy']:.4f} | {exp['notes'][:80]}")
    else:
        lines.append("- None yet")
    lines.append("")

    # Section 4: Unexplored Techniques
    lines.append("## Unexplored Techniques\n")
    tried_names = {e["name"] for e in experiments}
    unexplored = [t for t in EXPLORED_TECHNIQUES if t not in tried_names]
    if unexplored:
        for t in unexplored[:10]:
            lines.append(f"- `{t}`")
        if len(unexplored) > 10:
            lines.append(f"- ... and {len(unexplored) - 10} more")
    else:
        lines.append("- All listed techniques tried. Consult PROGRAM.md Tier 4/5 for novel ideas.")
    lines.append("")

    # Section 5: Memory Anomalies
    lines.append("## Memory Anomalies (MEMORY_LOG.md)\n")
    lines.append(_read_file(MEMORY_LOG_PATH))
    lines.append("")

    # Section 6: Research Notes
    lines.append("## Research Notes (NOTES.md)\n")
    lines.append(_read_file(NOTES_PATH))
    lines.append("")

    with open(CONTEXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"CONTEXT.md updated ({len(experiments)} experiments)")


if __name__ == "__main__":
    generate()
