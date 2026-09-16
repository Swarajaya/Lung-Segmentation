"""
Generate the qualitative visual results for a trained checkpoint.

Produces, from the held-out TEST split only:
  - a stratified result gallery covering SUCCESS / AVERAGE / DIFFICULT /
    FAILURE cases (not cherry-picked successes)
  - one detailed 6-panel figure per tier (image / GT / prediction / overlay /
    error map / probability map)
  - dedicated success_case, failure_case and error-map figures
  - a probability-map figure
  - an area-analysis CSV (GT vs predicted pixel area per image)

Usage:
    python scripts/generate_results.py --mode smoke_test
    python scripts/generate_results.py --mode full --experiment unet_aug_bce_dice --n-eval 200
"""
import os
import sys
import json
import argparse
import csv

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config, resolve_path
from src.evaluate import load_checkpoint
from src.predict import predict_from_config
from src.preprocessing import load_image_grayscale, load_mask, preprocess_mask
from src.visualization import (plot_sample_panel, plot_error_map, plot_probability_map,
                               plot_full_panel, plot_gallery)
from src.metrics import compute_all_metrics
from src.utils import ensure_dir, get_device, save_json


def stratify(candidates, n_per_tier):
    """
    Split GT-positive cases into four quality tiers by Dice.

    Using quartiles of the ACTUAL score distribution (rather than fixed
    thresholds) guarantees the gallery always contains the model's worst
    cases, whatever its overall quality — the gallery cannot degenerate into
    a highlight reel.
    """
    pos = sorted([c for c in candidates if c["gt_area"] > 0],
                 key=lambda c: c["dice"], reverse=True)
    if not pos:
        return {}
    n = len(pos)
    q = max(1, n // 4)
    return {
        "success": pos[:n_per_tier],
        "average": pos[q:q + n_per_tier] if n > q else [],
        "difficult": pos[2 * q:2 * q + n_per_tier] if n > 2 * q else [],
        "failure": pos[-n_per_tier:][::-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative result figures.")
    parser.add_argument("--mode", choices=["smoke_test", "full"], default="smoke_test")
    parser.add_argument("--experiment", default="unet_aug_bce_dice")
    parser.add_argument("--label", default=None,
                        help="Tag for output filenames (default: the --mode value).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-eval", type=int, default=60,
                        help="How many test images to score when ranking (CPU cost).")
    parser.add_argument("--n-panels", type=int, default=6)
    parser.add_argument("--n-per-tier", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    label = args.label or args.mode
    images_dir = resolve_path(cfg.data.images_dir)
    masks_dir = resolve_path(cfg.data.masks_dir)

    with open(resolve_path("data/splits.json")) as f:
        split = json.load(f)
    test_files = split["test_files"]

    ckpt_ref = f"outputs/checkpoints/{args.experiment}_best.pt"
    device = get_device(cfg.training.device)
    try:
        model, ckpt = load_checkpoint(ckpt_ref, cfg, device)
    except FileNotFoundError as e:
        print(f"{e}\nRun scripts/train.py first.")
        sys.exit(1)
    norm_stats = ckpt["normalize_stats"]

    figures_dir = resolve_path(cfg.paths.figures_dir)
    ensure_dir(figures_dir)

    cap = min(len(test_files), args.n_eval)
    print(f"Scoring {cap} test images with '{args.experiment}' to rank them...")
    candidates = []
    for fname in test_files[:cap]:
        img = load_image_grayscale(os.path.join(images_dir, fname))
        gt_bin = preprocess_mask(load_mask(os.path.join(masks_dir, fname)),
                                 cfg.data.image_size, cfg.data.mask_threshold)
        prob_map, pred_bin, resized = predict_from_config(
            model, norm_stats, device, img, cfg)
        m = compute_all_metrics(pred_bin, gt_bin)
        candidates.append({
            "filename": fname, "dice": m["dice"], "iou": m["iou"],
            "gt_area": int(gt_bin.sum()), "pred_area": int(pred_bin.sum()),
            "img01": resized.astype(np.float32) / 255.0,
            "gt": gt_bin, "pred": pred_bin, "prob": prob_map,
        })

    # ---- stratified gallery -------------------------------------------------
    tiers = stratify(candidates, args.n_per_tier)
    gallery_entries, tier_index = [], {}
    for tier, items in tiers.items():
        for c in items:
            gallery_entries.append({
                "image": c["img01"], "gt": c["gt"], "pred": c["pred"],
                "label": f"{tier.upper()}\n{c['filename']}\nDice={c['dice']:.3f}",
            })
        tier_index[tier] = [{"filename": c["filename"], "dice": c["dice"],
                             "iou": c["iou"], "gt_area_px": c["gt_area"],
                             "pred_area_px": c["pred_area"]} for c in items]

    # Always include empty-GT cases so the gallery represents the whole test set.
    empty_cases = [c for c in candidates if c["gt_area"] == 0][:args.n_per_tier]
    for c in empty_cases:
        gallery_entries.append({
            "image": c["img01"], "gt": c["gt"], "pred": c["pred"],
            "label": f"EMPTY GT\n{c['filename']}\npred={c['pred_area']} px",
        })
    tier_index["empty_gt"] = [{"filename": c["filename"], "dice": c["dice"],
                               "gt_area_px": 0, "pred_area_px": c["pred_area"]}
                              for c in empty_cases]

    if gallery_entries:
        gallery_path = os.path.join(figures_dir, f"{args.experiment}_{label}_gallery.png")
        plot_gallery(gallery_entries, gallery_path,
                     title=f"Result gallery — {args.experiment} ({label})\n"
                           f"stratified: success / average / difficult / failure / empty-GT")
        print(f"Wrote gallery: {gallery_path}")

    # ---- one detailed 6-panel figure per tier -------------------------------
    for tier, items in tiers.items():
        if not items:
            continue
        c = items[0]
        plot_full_panel(c["img01"], c["gt"], c["pred"], c["prob"],
                        os.path.join(figures_dir, f"{args.experiment}_{label}_{tier}_panel.png"),
                        title=f"{tier.upper()} case — {c['filename']} (Dice={c['dice']:.3f})")
    if empty_cases:
        c = empty_cases[0]
        plot_full_panel(c["img01"], c["gt"], c["pred"], c["prob"],
                        os.path.join(figures_dir, f"{args.experiment}_{label}_empty_gt_panel.png"),
                        title=f"EMPTY GROUND TRUTH — {c['filename']} "
                              f"(predicted {c['pred_area']} foreground px)")

    # ---- evenly-spread representative 5-panel figures -----------------------
    ranked = sorted(candidates, key=lambda c: c["dice"], reverse=True)
    step = max(1, len(ranked) // max(1, args.n_panels))
    for i, c in enumerate(ranked[::step][:args.n_panels]):
        plot_sample_panel(c["img01"], c["gt"], c["pred"],
                          os.path.join(figures_dir,
                                       f"{label}_sample_panel_{i}_{os.path.splitext(c['filename'])[0]}.png"),
                          title=f"{c['filename']}  (Dice={c['dice']:.3f})")

    # ---- dedicated success / failure / probability figures ------------------
    non_trivial = [c for c in candidates if c["gt_area"] > 0]
    if non_trivial:
        best = max(non_trivial, key=lambda c: c["dice"])
        worst = min(non_trivial, key=lambda c: c["dice"])
        plot_sample_panel(best["img01"], best["gt"], best["pred"],
                          os.path.join(figures_dir, f"{label}_success_case.png"),
                          title=f"Success case: {best['filename']} (Dice={best['dice']:.3f})")
        plot_sample_panel(worst["img01"], worst["gt"], worst["pred"],
                          os.path.join(figures_dir, f"{label}_failure_case.png"),
                          title=f"Failure case: {worst['filename']} (Dice={worst['dice']:.3f})")
        plot_error_map(worst["img01"], worst["gt"], worst["pred"],
                       os.path.join(figures_dir, f"{label}_failure_case_error_map.png"),
                       title=f"Prediction error map: {worst['filename']}")
        plot_probability_map(best["img01"], best["prob"],
                             os.path.join(figures_dir, f"{label}_probability_map_example.png"))

    # ---- area analysis ------------------------------------------------------
    area_csv = resolve_path(f"outputs/metrics/area_analysis_{label}.csv")
    ensure_dir(os.path.dirname(area_csv))
    with open(area_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "gt_area_px", "pred_area_px", "abs_area_diff_px",
                    "relative_area_diff", "dice", "iou"])
        for c in candidates:
            d = abs(c["gt_area"] - c["pred_area"])
            rel = (d / c["gt_area"]) if c["gt_area"] > 0 else ("n/a (empty GT)" if c["pred_area"] == 0 else "inf")
            w.writerow([c["filename"], c["gt_area"], c["pred_area"], d, rel,
                        round(c["dice"], 4), round(c["iou"], 4)])

    save_json({"experiment": args.experiment, "mode": args.mode, "label": label,
               "n_images_scored": len(candidates), "tiers": tier_index},
              resolve_path(f"outputs/metrics/{args.experiment}_{label}_gallery_index.json"))

    print(f"Wrote figures to {figures_dir}")
    print(f"Wrote area analysis to {area_csv}")
    print("NOTE: pixel area is NOT physical lung/tumor area — this dataset has no "
          "pixel-spacing (mm/pixel) metadata, so only pixel counts are reported.")


if __name__ == "__main__":
    main()
