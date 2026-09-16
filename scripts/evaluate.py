"""
Evaluate trained (or smoke-test) checkpoints on the held-out TEST split and
write a results table (CSV + markdown) containing only real, computed metrics.

Reports three views per experiment — all images, GT-positive images only, and
empty-GT images only — because ~16% of this dataset's masks are empty and
score a free Dice of 1.0 when the model correctly predicts nothing. The
"positive only" column is the one that measures segmentation quality; the
"all" column is what you get on the raw test set. Neither is hidden.

Usage:
    python scripts/evaluate.py --mode smoke_test
    python scripts/evaluate.py --mode full
    python scripts/evaluate.py --mode full --threshold 0.4 --no-surface
"""
import os
import sys
import json
import argparse
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config, resolve_path
from src.evaluate import evaluate_checkpoint, resolve_checkpoint_path
from src.utils import ensure_dir, save_json


def _fmt(v, nd=4):
    return round(v, nd) if isinstance(v, (int, float)) else ("-" if v is None else v)


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on the test split.")
    parser.add_argument("--mode", choices=["smoke_test", "full"], default="smoke_test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap test samples evaluated (useful on CPU-only smoke runs).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Probability threshold (default: evaluation.threshold in config).")
    parser.add_argument("--label", default=None,
                        help="Tag for input/output filenames (default: the --mode value).")
    parser.add_argument("--no-surface", action="store_true",
                        help="Skip HD95/ASSD (faster).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    label = args.label or args.mode
    with open(resolve_path("data/splits.json")) as f:
        split = json.load(f)
    test_files = split["test_files"]

    results_path = resolve_path(f"outputs/metrics/training_results_{label}.json")
    if not os.path.exists(results_path):
        print(f"No {results_path} found. Run scripts/train.py --mode {args.mode} first.")
        sys.exit(1)
    with open(results_path) as f:
        training_results = json.load(f)

    rows = []
    for exp in training_results["experiments"]:
        exp_name = exp["experiment_name"]
        try:
            ckpt = resolve_checkpoint_path(exp.get("best_checkpoint") or exp_name)
        except FileNotFoundError as e:
            print(f"[{exp_name}] {e}")
            rows.append({"experiment": exp_name, "note":
                         "No checkpoint found — training did not complete in this environment."})
            continue

        max_samples = args.max_samples or (32 if args.mode == "smoke_test" else None)
        eval_result = evaluate_checkpoint(
            cfg, ckpt, test_files, split_name="test", max_samples=max_samples,
            threshold=args.threshold,
            include_surface=(False if args.no_surface else None))

        sg = eval_result["subgroup_metrics"]
        a, pos, emp = sg["all"], sg["positive_only"], sg["empty_gt_only"]

        # Prefer the note recorded by the training run itself, so a
        # capacity-reduced run can never be labelled as a full-quality one.
        note = training_results.get("note") or (
            "SMOKE TEST (tiny data subset, 2 epochs) — proves the pipeline runs "
            "end-to-end. NOT a measure of model quality."
            if args.mode == "smoke_test" else
            "Training run at the settings in the config file used for this run.")

        rows.append({
            "experiment": exp_name,
            "model": exp.get("model", "unet"),
            "augmentation": exp.get("augmentation"),
            "loss": exp.get("loss"),
            "dice_all": _fmt(a["dice"]), "iou_all": _fmt(a["iou"]),
            "dice_positive_only": _fmt(pos["dice"]), "iou_positive_only": _fmt(pos["iou"]),
            "precision": _fmt(a["precision"]), "recall": _fmt(a["recall"]),
            "specificity": _fmt(a["specificity"]), "f1": _fmt(a["f1"]),
            "foreground_coverage": _fmt(a["foreground_coverage"]),
            "hd95_px": _fmt(pos["hd95"], 3), "assd_px": _fmt(pos["assd"], 3),
            "n_hd95_defined": pos.get("n_hd95_defined", 0),
            "dice_empty_gt_only": _fmt(emp["dice"]),
            "n_all": sg["n_all"], "n_positive_gt": sg["n_positive_gt"],
            "n_empty_gt": sg["n_empty_gt"],
            "tp": a["tp"], "tn": a["tn"], "fp": a["fp"], "fn": a["fn"],
            "note": note,
        })

        save_json(eval_result, resolve_path(
            f"outputs/metrics/{exp_name}_{label}_test_eval.json"))

    fieldnames = ["experiment", "model", "augmentation", "loss",
                  "dice_all", "iou_all", "dice_positive_only", "iou_positive_only",
                  "precision", "recall", "specificity", "f1", "foreground_coverage",
                  "hd95_px", "assd_px", "n_hd95_defined", "dice_empty_gt_only",
                  "n_all", "n_positive_gt", "n_empty_gt", "tp", "tn", "fp", "fn", "note"]

    out_csv = resolve_path(f"outputs/metrics/results_table_{label}.csv")
    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    out_md = resolve_path(f"outputs/metrics/results_table_{label}.md")
    with open(out_md, "w") as f:
        f.write(f"# Results table ({label})\n\n")
        f.write("All numbers below were computed by `scripts/evaluate.py` on the "
                "held-out test split. `dice_all` includes empty-ground-truth "
                "slices (which score 1.0 when correctly predicted empty); "
                "`dice_positive_only` covers only slices that actually contain "
                "foreground and is the figure that measures segmentation "
                "quality. HD95/ASSD are in pixels and are averaged over the "
                "cases where they are defined.\n\n")
        f.write("| Experiment | Model | Aug | Loss | Dice (all) | Dice (GT+) | "
                "IoU (GT+) | Precision | Recall | Specificity | HD95 px | ASSD px | Note |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['experiment']} | {r.get('model','-')} | {r.get('augmentation','-')} | "
                    f"{r.get('loss','-')} | {r.get('dice_all','-')} | "
                    f"{r.get('dice_positive_only','-')} | {r.get('iou_positive_only','-')} | "
                    f"{r.get('precision','-')} | {r.get('recall','-')} | "
                    f"{r.get('specificity','-')} | {r.get('hd95_px','-')} | "
                    f"{r.get('assd_px','-')} | {r.get('note','')} |\n")

    print(f"Wrote {out_csv}\nWrote {out_md}")
    for r in rows:
        print(json.dumps(r, default=str))


if __name__ == "__main__":
    main()
