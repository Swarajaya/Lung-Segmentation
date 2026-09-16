"""
Run the experiment matrix defined in config.yaml.

Usage
-----
    python scripts/train.py --mode smoke_test            # CPU-safe pipeline check
    python scripts/train.py --mode full                  # GPU machine
    python scripts/train.py --mode full --include-optional
    python scripts/train.py --mode full --experiment unet_aug_bce_dice
    python scripts/train.py --mode full --experiment unet_aug_bce_dice \
        --resume outputs/checkpoints/unet_aug_bce_dice_last.pt

The smoke test uses a small, capped subset of data and 2 epochs purely to
prove the pipeline (data loading -> model -> loss -> backward -> checkpoint ->
evaluation) is correct end-to-end. It is NOT a measure of real model quality,
and its results are written to separate files (`*_smoke_test.*`) so a smoke
number can never be mistaken for a trained number.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config, resolve_path
from src.utils import ensure_dir, save_json, describe_hardware
from src.train import train_model
from src.visualization import plot_training_curves, plot_lr_curve


def main():
    parser = argparse.ArgumentParser(description="Train the segmentation experiment matrix.")
    parser.add_argument("--mode", choices=["smoke_test", "full"], default="smoke_test")
    parser.add_argument("--config", default=None, help="Path to a config YAML (default: config.yaml)")
    parser.add_argument("--experiment", default=None,
                        help="Run only this experiment name from config.experiments")
    parser.add_argument("--include-optional", action="store_true",
                        help="Also run experiments marked 'enabled: false' in config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Resume from a checkpoint (only valid with --experiment)")
    parser.add_argument("--label", default=None,
                        help="Tag for output filenames (default: the --mode value). Use this "
                             "to keep runs separate, e.g. --label cpu_real.")
    parser.add_argument("--out", default=None, help="Override the results JSON output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    label = args.label or args.mode

    hw = describe_hardware()
    print("Hardware detected:", json.dumps(hw))
    if args.mode == "full" and not hw["cuda_available"]:
        print("\nWARNING: --mode full requested but no CUDA device is present. "
              "Full training on CPU will be extremely slow. See README "
              "'Full training on a GPU' for the recommended command.\n")

    split_path = resolve_path("data/splits.json")
    if not os.path.exists(split_path):
        print("No data/splits.json found — run scripts/split_dataset.py first.")
        sys.exit(1)
    with open(split_path) as f:
        split = json.load(f)

    experiments = list(cfg.experiments)
    if args.experiment:
        experiments = [e for e in experiments if e["name"] == args.experiment]
        if not experiments:
            print(f"No experiment named '{args.experiment}' in config.yaml. "
                  f"Available: {[e['name'] for e in cfg.experiments]}")
            sys.exit(1)
    elif not args.include_optional:
        experiments = [e for e in experiments if e.get("enabled", True)]

    if args.resume and len(experiments) != 1:
        print("--resume requires exactly one --experiment.")
        sys.exit(1)

    print(f"Running {len(experiments)} experiment(s): {[e['name'] for e in experiments]}")

    results = {"mode": args.mode, "label": label, "hardware": hw,
               "config_file": args.config or "config.yaml",
               "data": {"image_size": cfg.data.image_size,
                        "n_train_files": len(split["train_files"]),
                        "n_val_files": len(split["val_files"]),
                        "n_train_subjects": len(split["train_subjects"]),
                        "n_val_subjects": len(split["val_subjects"])},
               "model_base_channels": cfg.model.base_channels,
               "experiments": []}

    for exp in experiments:
        exp_name = exp["name"]
        print(f"\n=== Running experiment: {exp_name} (mode={args.mode}) ===")
        result = train_model(
            cfg, split, mode=args.mode, experiment_name=exp_name,
            augmentation_override=exp.get("augmentation"),
            loss_override=exp.get("loss"),
            model_override=exp.get("model"),
            resume_from=args.resume,
        )
        results["experiments"].append(result)

        figures_dir = resolve_path(cfg.paths.figures_dir)
        ensure_dir(figures_dir)
        curve_path = os.path.join(figures_dir, f"{exp_name}_{label}_training_curves.png")
        plot_training_curves(result["history"], curve_path,
                             title_prefix=f"{exp_name} ({label}) - ")
        lr_path = os.path.join(figures_dir, f"{exp_name}_{label}_learning_rate.png")
        plot_lr_curve(result["history"], lr_path,
                      title=f"{exp_name} ({label}) — learning rate")
        print(f"Saved training curves to {curve_path}")
        print(f"Saved learning-rate plot to {lr_path}")

    out_path = args.out or resolve_path(f"outputs/metrics/training_results_{label}.json")
    ensure_dir(os.path.dirname(out_path))
    save_json(results, out_path)
    print(f"\nWrote experiment results to {out_path}")


if __name__ == "__main__":
    main()
