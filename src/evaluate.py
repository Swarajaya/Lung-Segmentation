"""
Evaluate a trained checkpoint on a held-out (val/test) split.

Reports three views of every metric — all images, ground-truth-positive images
only, and empty-ground-truth images only — because on this dataset ~16% of
masks are empty and an empty-GT/empty-prediction pair scores a free Dice of
1.0. Averaging those into a single headline number would overstate
segmentation quality. See src/metrics.py for the full rationale; no case is
ever silently dropped.
"""
import os
import torch
from torch.utils.data import DataLoader

from .dataset import LungSegmentationDataset, subject_id_from_filename
from .models import build_model
from .metrics import compute_all_metrics, aggregate_by_subgroup
from .utils import get_device
from .config import resolve_path, get_project_root


def resolve_checkpoint_path(path_or_name: str) -> str:
    """
    Resolve a checkpoint reference to an existing absolute path.

    Result files written on one machine may record an absolute path that does
    not exist on another. We therefore accept an absolute path, a path
    relative to the project root, or a bare experiment name, and fall back to
    `outputs/checkpoints/<basename>` — which keeps older result files usable
    after the project is copied to a different machine.
    """
    if not path_or_name:
        raise FileNotFoundError("No checkpoint path given.")
    if os.path.isabs(path_or_name) and os.path.exists(path_or_name):
        return path_or_name

    candidates = [
        os.path.join(get_project_root(), path_or_name),
        resolve_path(os.path.join("outputs/checkpoints", os.path.basename(path_or_name))),
        resolve_path(os.path.join("outputs/checkpoints", f"{path_or_name}_best.pt")),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Checkpoint '{path_or_name}' not found. Looked in: {candidates}")


def load_checkpoint(ckpt_path, cfg, device=None):
    """Load a checkpoint, rebuilding the architecture it was trained with."""
    device = device or get_device(cfg.training.device)
    ckpt_path = resolve_checkpoint_path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Older checkpoints predate the model registry; they are plain U-Nets.
    model_name = ckpt.get("model_name", "unet")
    model = build_model(cfg, name_override=model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def evaluate_checkpoint(cfg, ckpt_path, files, split_name="test", max_samples=None,
                        batch_size=8, threshold=None, include_surface=None):
    """Run full evaluation of one checkpoint over `files`."""
    device = get_device(cfg.training.device)
    model, ckpt = load_checkpoint(ckpt_path, cfg, device)
    stats = ckpt.get("normalize_stats", {"mean": 0.5, "std": 0.25})
    mean, std = stats["mean"], stats["std"]

    threshold = cfg.evaluation.threshold if threshold is None else threshold
    if include_surface is None:
        include_surface = bool(cfg.evaluation.get("compute_surface_metrics", True))

    images_dir = resolve_path(cfg.data.images_dir)
    masks_dir = resolve_path(cfg.data.masks_dir)

    ds = LungSegmentationDataset.from_config(
        cfg, files, images_dir, masks_dir, transform=None,
        normalize_stats=(mean, std), max_samples=max_samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    per_sample_metrics, area_records = [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            fnames = batch["filename"]

            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds_bin = (probs >= threshold).astype("uint8")
            gts_bin = masks.cpu().numpy().astype("uint8")

            for i in range(preds_bin.shape[0]):
                m = compute_all_metrics(preds_bin[i, 0], gts_bin[i, 0],
                                        include_surface=include_surface)
                m["filename"] = fnames[i]
                m["subject_id"] = subject_id_from_filename(fnames[i])
                m["mean_pred_prob"] = float(probs[i, 0].mean())
                per_sample_metrics.append(m)

                gt_area = int(gts_bin[i, 0].sum())
                pred_area = int(preds_bin[i, 0].sum())
                abs_diff = abs(gt_area - pred_area)
                area_records.append({
                    "filename": fnames[i],
                    "gt_area_px": gt_area,
                    "pred_area_px": pred_area,
                    "abs_area_diff_px": abs_diff,
                    "relative_area_diff": (abs_diff / gt_area) if gt_area > 0
                                          else (None if pred_area == 0 else float("inf")),
                })

    subgroups = aggregate_by_subgroup(per_sample_metrics)
    agg = subgroups["all"]
    agg["split_name"] = split_name
    # Stored RELATIVE to the project root so results files remain valid (and
    # free of machine-specific paths) when the project is copied elsewhere.
    agg["checkpoint"] = os.path.relpath(resolve_checkpoint_path(ckpt_path), get_project_root())
    agg["threshold"] = threshold

    return {
        "aggregate_metrics": agg,
        "subgroup_metrics": subgroups,
        "per_sample_metrics": per_sample_metrics,
        "area_analysis": area_records,
        "evaluation_config": {
            "threshold": threshold,
            "surface_metrics_computed": include_surface,
            "n_evaluated": len(per_sample_metrics),
            "split": split_name,
        },
    }
