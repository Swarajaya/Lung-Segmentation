"""
Pixel-level segmentation metrics computed from confusion matrix components.

All metrics are computed from actual predicted vs. ground-truth binary masks
— nothing here is hard-coded or fabricated. Edge cases (e.g. an entirely
empty ground-truth AND empty prediction) are handled explicitly to avoid
division by zero, and are documented inline.
"""
import numpy as np
import torch

try:
    from scipy.ndimage import distance_transform_edt, binary_erosion
    _SCIPY_AVAILABLE = True
except ImportError:  # surface metrics degrade to None rather than crashing
    _SCIPY_AVAILABLE = False


def confusion_components(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """Return TP, TN, FP, FN as plain Python ints for a pair of binary masks."""
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)
    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    tn = int(np.logical_and(~pred_bin, ~gt_bin).sum())
    fp = int(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = int(np.logical_and(~pred_bin, gt_bin).sum())
    return tp, tn, fp, fn


def dice_from_confusion(tp, fp, fn, eps=1e-8):
    denom = 2 * tp + fp + fn
    if denom == 0:
        # Both prediction and ground truth are empty -> perfect agreement.
        return 1.0
    return (2 * tp) / (denom + eps)


def iou_from_confusion(tp, fp, fn, eps=1e-8):
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return tp / (denom + eps)


def precision_from_confusion(tp, fp, eps=1e-8):
    if tp + fp == 0:
        return 1.0 if tp == 0 else 0.0
    return tp / (tp + fp + eps)


def recall_from_confusion(tp, fn, eps=1e-8):
    if tp + fn == 0:
        return 1.0 if tp == 0 else 0.0
    return tp / (tp + fn + eps)


def specificity_from_confusion(tn, fp, eps=1e-8):
    if tn + fp == 0:
        return 1.0
    return tn / (tn + fp + eps)


def f1_from_confusion(tp, fp, fn, eps=1e-8):
    p = precision_from_confusion(tp, fp, eps)
    r = recall_from_confusion(tp, fn, eps)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_all_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray,
                        include_surface: bool = False) -> dict:
    """
    Compute the full metric suite for a single (pred, gt) binary mask pair.

    `include_surface=True` additionally computes HD95 and ASSD, which require
    two distance transforms per sample and so are off by default in the
    training loop (where they would dominate epoch time) and on during
    final evaluation.
    """
    tp, tn, fp, fn = confusion_components(pred_bin, gt_bin)
    dice = dice_from_confusion(tp, fp, fn)
    iou = iou_from_confusion(tp, fp, fn)
    precision = precision_from_confusion(tp, fp)
    recall = recall_from_confusion(tp, fn)
    specificity = specificity_from_confusion(tn, fp)
    f1 = f1_from_confusion(tp, fp, fn)
    result = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "dice": dice, "iou": iou,
        "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1,
        # Foreground coverage: what fraction of the image each mask occupies,
        # and what fraction of the GT foreground the prediction covered.
        "gt_foreground_fraction": float(np.asarray(gt_bin).astype(bool).mean()),
        "pred_foreground_fraction": float(np.asarray(pred_bin).astype(bool).mean()),
        "foreground_coverage": (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if tp == 0 else 0.0),
    }
    if include_surface:
        result.update(compute_surface_metrics(pred_bin, gt_bin))
    return result


def logits_to_binary(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return (probs >= threshold).astype(np.uint8)


def aggregate_metrics(metric_dicts: list) -> dict:
    """Average a list of per-sample metric dicts (dice/iou/precision/recall/specificity/f1),
    and sum the confusion components."""
    keys_avg = ["dice", "iou", "precision", "recall", "specificity", "f1",
                "foreground_coverage"]
    keys_sum = ["tp", "tn", "fp", "fn"]
    keys_optional_avg = ["hd95", "assd"]

    out = {}
    for k in keys_avg:
        vals = [m[k] for m in metric_dicts if k in m]
        out[k] = float(np.mean(vals)) if vals else 0.0
    for k in keys_sum:
        out[k] = int(np.sum([m[k] for m in metric_dicts])) if metric_dicts else 0

    # Surface metrics are undefined for some samples (see notes above): average
    # only the defined ones and record how many contributed.
    for k in keys_optional_avg:
        vals = [m[k] for m in metric_dicts if m.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
        out[f"n_{k}_defined"] = len(vals)

    out["n_samples"] = len(metric_dicts)
    return out


# ---------------------------------------------------------------------------
# Surface / boundary distance metrics
#
# Dice and IoU are overlap metrics: they say how much of the region is right,
# but a prediction can have a decent Dice while its BOUNDARY is badly placed
# (e.g. a smooth blob approximating a jagged lesion). Surface distance metrics
# measure boundary quality directly, in pixels.
#
#   HD95 - the 95th percentile of the symmetric set of distances from each
#          boundary point of one mask to the nearest boundary point of the
#          other. The 95th percentile rather than the true maximum (Hausdorff)
#          is standard in medical segmentation because a single stray
#          mislabelled pixel would otherwise dominate the score.
#
#   ASSD - Average Symmetric Surface Distance: the mean of those same
#          distances. Sensitive to overall boundary displacement rather than
#          to worst-case outliers.
#
# Both are reported in PIXELS, not millimetres: this dataset ships as PNG
# slices with no pixel-spacing (mm/pixel) metadata, so any millimetre figure
# would be invented. This is stated in the report rather than glossed over.
#
# Undefined cases: a distance between surfaces requires both surfaces to
# exist. If exactly one of (prediction, ground truth) is empty the distance is
# genuinely undefined/infinite, and if both are empty there is no boundary at
# all. We return None in both cases and EXCLUDE them from the surface-metric
# average, while reporting how many cases were excluded — rather than
# substituting 0 (which would flatter the model) or the image diagonal (which
# would punish it arbitrarily).
# ---------------------------------------------------------------------------


def _surface_points(mask_bin: np.ndarray) -> np.ndarray:
    """Boolean array marking the inner boundary pixels of a binary mask."""
    mask_bin = mask_bin.astype(bool)
    if not mask_bin.any():
        return np.zeros_like(mask_bin, dtype=bool)
    eroded = binary_erosion(mask_bin, structure=np.ones((3, 3)), border_value=0)
    return mask_bin & ~eroded


def surface_distances(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """
    Return the concatenated symmetric surface distance array (in pixels),
    or None when either mask is empty (distance undefined).
    """
    if not _SCIPY_AVAILABLE:
        return None
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)
    if not pred_bin.any() or not gt_bin.any():
        return None

    pred_surf = _surface_points(pred_bin)
    gt_surf = _surface_points(gt_bin)
    if not pred_surf.any() or not gt_surf.any():
        return None

    # distance_transform_edt gives, for each pixel, the distance to the
    # nearest ZERO pixel — so invert the surface to get distance-to-surface.
    dt_to_gt = distance_transform_edt(~gt_surf)
    dt_to_pred = distance_transform_edt(~pred_surf)

    d_pred_to_gt = dt_to_gt[pred_surf]
    d_gt_to_pred = dt_to_pred[gt_surf]
    return np.concatenate([d_pred_to_gt, d_gt_to_pred])


def hd95(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """95th-percentile symmetric Hausdorff distance in pixels, or None."""
    d = surface_distances(pred_bin, gt_bin)
    if d is None or d.size == 0:
        return None
    return float(np.percentile(d, 95))


def assd(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """Average symmetric surface distance in pixels, or None."""
    d = surface_distances(pred_bin, gt_bin)
    if d is None or d.size == 0:
        return None
    return float(np.mean(d))


def compute_surface_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict:
    """Both surface metrics in one distance-transform pass."""
    d = surface_distances(pred_bin, gt_bin)
    if d is None or d.size == 0:
        return {"hd95": None, "assd": None}
    return {"hd95": float(np.percentile(d, 95)), "assd": float(np.mean(d))}


# ---------------------------------------------------------------------------
# Subgroup aggregation
#
# Reporting a single average over all images is misleading on this dataset:
# ~16% of ground-truth masks are empty, and an empty-GT/empty-pred pair scores
# a perfect Dice of 1.0 by definition. Those free 1.0s inflate the headline
# mean without the model having segmented anything. We therefore always report
# three views, and the report explains the difference:
#
#   all            - every test image, empty-GT cases included (scored as
#                    described above). This is the honest "what happens if you
#                    feed the model the whole test set" number.
#   positive_only  - only images whose GROUND TRUTH contains foreground. This
#                    is the number that actually measures segmentation quality.
#   empty_gt_only  - only images whose ground truth is empty. Here Dice is 1.0
#                    exactly when the model correctly predicts nothing, so this
#                    subgroup measures the model's false-positive behaviour on
#                    negative slices.
#
# Empty-GT cases are never silently excluded — they are moved into their own
# reported subgroup.
# ---------------------------------------------------------------------------


def aggregate_by_subgroup(metric_dicts: list) -> dict:
    """Aggregate a list of per-sample metric dicts into the three views above."""
    all_m = list(metric_dicts)
    positive = [m for m in all_m if (m["tp"] + m["fn"]) > 0]
    empty_gt = [m for m in all_m if (m["tp"] + m["fn"]) == 0]
    return {
        "all": aggregate_metrics(all_m),
        "positive_only": aggregate_metrics(positive),
        "empty_gt_only": aggregate_metrics(empty_gt),
        "n_all": len(all_m),
        "n_positive_gt": len(positive),
        "n_empty_gt": len(empty_gt),
    }
