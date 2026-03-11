"""CLI: log experiment result to DB and regenerate CONTEXT.md."""
import argparse
import sys
from db import insert_experiment, get_best_kept_accuracy, set_kept
import context_gen


def main():
    parser = argparse.ArgumentParser(description="Log an IFEval fine-tuning experiment result.")
    parser.add_argument("--name",       required=True,  help="Experiment name, e.g. lora_r16")
    parser.add_argument("--accuracy",   required=True,  type=float, help="prompt_level_strict_acc (0.0–1.0)")
    parser.add_argument("--notes",      default="",     help="What you changed and observed")
    parser.add_argument("--hypothesis", default="",     help="Why you expected this to work")
    args = parser.parse_args()

    BASELINE = 0.612

    # Insert experiment
    exp_id = insert_experiment(
        name=args.name,
        accuracy=args.accuracy,
        notes=args.notes,
        hypothesis=args.hypothesis,
    )

    # Check if this is the new best
    best = get_best_kept_accuracy()
    is_new_best = best is None or args.accuracy > best

    kept = False
    if is_new_best:
        set_kept(exp_id)
        kept = True

    # Regenerate CONTEXT.md
    context_gen.generate()

    # Print summary
    delta = args.accuracy - BASELINE
    delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
    print(f"\n{'='*60}")
    print(f"  Experiment : {args.name}")
    print(f"  Accuracy   : {args.accuracy:.4f}  (baseline {BASELINE:.3f}, delta {delta_str})")
    print(f"  Kept       : {'YES ✓' if kept else 'NO  ✗'}")
    if kept:
        print(f"  New best   : {args.accuracy:.4f}")
    else:
        print(f"  Best so far: {best:.4f}")
    print(f"{'='*60}\n")

    if kept:
        print(f"Suggested commit:")
        print(f"  git add -A && git commit -m 'exp: {args.name} acc={args.accuracy:.4f} (delta={delta_str})'")


if __name__ == "__main__":
    main()
