"""
Run inference on a single image or a directory of images.

Usage:
    python scripts/predict.py --input path/to/slice.png
    python scripts/predict.py --input data/lung-cancer-vision-v1/images --limit 10
    python scripts/predict.py --input slice.png --experiment unet_aug_bce_dice \
        --threshold 0.4 --output-dir outputs/predictions --save-figure

Outputs (to outputs/predictions/ by default):
    <name>_mask.png       binary predicted mask (0/255)
    <name>_prob.npy       raw sigmoid probability map (float32)
    <name>_panel.png      visual panel (with --save-figure)
    predictions.json      per-image statistics

If a matching ground-truth mask exists in the configured masks directory, the
metrics for that image are computed and included — otherwise only descriptive
statistics are reported. Nothing is estimated when ground truth is absent.
"""
import os
import sys
import json
import argparse

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config, resolve_path
from src.predict import load_model_for_inference, predict_from_config, mask_statistics
from src.preprocessing import load_image_grayscale, load_mask, preprocess_mask
from src.metrics import compute_all_metrics
from src.visualization import plot_full_panel
from src.utils import ensure_dir, save_json

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def main():
    parser = argparse.ArgumentParser(description="Run segmentation inference on image(s).")
    parser.add_argument("--input", required=True, help="Image file or directory of images")
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment", default="unet_aug_bce_dice",
                        help="Experiment name whose *_best.pt checkpoint to use")
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit checkpoint path (overrides --experiment)")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max images when --input is a directory")
    parser.add_argument("--save-figure", action="store_true", help="Also save a visual panel per image")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_ref = args.checkpoint or f"outputs/checkpoints/{args.experiment}_best.pt"

    try:
        model, norm_stats, device = load_model_for_inference(ckpt_ref, cfg)
    except FileNotFoundError as e:
        print(f"{e}\nTrain a model first:  python scripts/train.py --mode smoke_test")
        sys.exit(1)

    if os.path.isdir(args.input):
        files = sorted(f for f in os.listdir(args.input) if f.lower().endswith(IMAGE_EXTS))
        if args.limit:
            files = files[:args.limit]
        paths = [os.path.join(args.input, f) for f in files]
    elif os.path.isfile(args.input):
        paths = [args.input]
    else:
        print(f"Input not found: {args.input}")
        sys.exit(1)

    if not paths:
        print(f"No images found in {args.input}")
        sys.exit(1)

    out_dir = args.output_dir or resolve_path(cfg.paths.predictions_dir)
    ensure_dir(out_dir)
    masks_dir = resolve_path(cfg.data.masks_dir)

    records = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        img = load_image_grayscale(path)
        prob_map, pred_mask, resized = predict_from_config(
            model, norm_stats, device, img, cfg, threshold=args.threshold)

        Image.fromarray((pred_mask * 255).astype(np.uint8)).save(
            os.path.join(out_dir, f"{name}_mask.png"))
        np.save(os.path.join(out_dir, f"{name}_prob.npy"), prob_map.astype(np.float32))

        rec = {"input": path, "source_shape": list(img.shape)}
        rec.update(mask_statistics(pred_mask, prob_map))

        gt_path = os.path.join(masks_dir, os.path.basename(path))
        gt_bin = None
        if os.path.exists(gt_path):
            gt_bin = preprocess_mask(load_mask(gt_path), cfg.data.image_size,
                                     cfg.data.mask_threshold)
            m = compute_all_metrics(pred_mask, gt_bin, include_surface=True)
            rec["metrics_vs_ground_truth"] = {
                k: m[k] for k in ("dice", "iou", "precision", "recall",
                                  "specificity", "f1", "hd95", "assd", "tp", "fp", "fn")}
        else:
            rec["metrics_vs_ground_truth"] = None

        if args.save_figure:
            fig_path = os.path.join(out_dir, f"{name}_panel.png")
            gt_for_plot = gt_bin if gt_bin is not None else np.zeros_like(pred_mask)
            title = name + (f"  (Dice={rec['metrics_vs_ground_truth']['dice']:.3f})"
                            if rec["metrics_vs_ground_truth"] else "  (no ground truth available)")
            plot_full_panel(resized.astype(np.float32) / 255.0, gt_for_plot,
                            pred_mask, prob_map, fig_path, title=title)
            rec["figure"] = fig_path

        records.append(rec)
        fg = rec["foreground_pixels"]
        dice_txt = (f" dice={rec['metrics_vs_ground_truth']['dice']:.4f}"
                    if rec["metrics_vs_ground_truth"] else "")
        print(f"{os.path.basename(path)}: foreground={fg} px "
              f"({rec['foreground_percent']:.3f}%) components={rec['n_components']}{dice_txt}")

    save_json({"checkpoint": ckpt_ref,
               "threshold": args.threshold if args.threshold is not None else cfg.evaluation.threshold,
               "n_images": len(records), "predictions": records},
              os.path.join(out_dir, "predictions.json"))
    print(f"\nWrote {len(records)} prediction(s) to {out_dir}")
    print("NOTE: areas are in PIXELS. This dataset carries no pixel-spacing "
          "(mm/pixel) metadata, so physical area/volume cannot be derived.")


if __name__ == "__main__":
    main()
