"""
Run the automated error analysis on a completed evaluation.

Usage:
    python scripts/error_analysis.py --mode smoke_test
    python scripts/error_analysis.py --mode full --experiment unet_aug_bce_dice

Reads outputs/metrics/<experiment>_<mode>_test_eval.json (produced by
scripts/evaluate.py) and writes:
    outputs/metrics/<experiment>_<mode>_error_analysis.md
    outputs/metrics/<experiment>_<mode>_error_analysis.csv
    outputs/metrics/<experiment>_<mode>_error_analysis.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config, resolve_path
from src.error_analysis import analyze, write_csv, write_markdown
from src.utils import save_json


def main():
    parser = argparse.ArgumentParser(description="Automated segmentation error analysis.")
    parser.add_argument("--mode", choices=["smoke_test", "full"], default="smoke_test")
    parser.add_argument("--experiment", default=None,
                        help="Experiment name (default: every *_<mode>_test_eval.json found)")
    parser.add_argument("--label", default=None,
                        help="Tag for input/output filenames (default: the --mode value).")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    label = args.label or args.mode
    metrics_dir = resolve_path(cfg.paths.metrics_dir)

    suffix = f"_{label}_test_eval.json"
    if args.experiment:
        targets = [f"{args.experiment}{suffix}"]
    else:
        targets = sorted(f for f in os.listdir(metrics_dir) if f.endswith(suffix))

    if not targets:
        print(f"No *{suffix} files in {metrics_dir}. "
              f"Run scripts/evaluate.py --mode {args.mode} first.")
        sys.exit(1)

    for fname in targets:
        path = os.path.join(metrics_dir, fname)
        if not os.path.exists(path):
            print(f"Missing {path} — skipping.")
            continue
        exp_name = fname[:-len(suffix)]
        with open(path) as f:
            eval_result = json.load(f)

        analysis = analyze(eval_result["per_sample_metrics"], cfg)

        base = os.path.join(metrics_dir, f"{exp_name}_{label}_error_analysis")
        write_markdown(analysis, base + ".md", exp_name, label)
        write_csv(analysis, base + ".csv")
        save_json({k: v for k, v in analysis.items() if k != "per_sample"}, base + ".json")

        print(f"\n=== {exp_name} ({label}) ===")
        print(f"  analysed {analysis['n_samples']} images "
              f"({analysis['n_gt_positive']} with foreground, {analysis['n_gt_empty']} empty GT)")
        for k, v in analysis["category_counts"].items():
            print(f"  {k:<22} {v}")
        print(f"  Wrote {base}.md / .csv / .json")


if __name__ == "__main__":
    main()
