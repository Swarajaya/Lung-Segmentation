"""Single-image inference pipeline (used by scripts/predict.py and the Streamlit app)."""
import numpy as np
import torch

from .preprocessing import resize_image, normalize_image, apply_clahe
from .models import build_model
from .utils import get_device


def load_model_for_inference(ckpt_path, cfg, device=None):
    """
    Load a checkpoint for inference.

    The normalization statistics are read back FROM THE CHECKPOINT, not
    recomputed: inference must use exactly the statistics the model was
    trained with, or every prediction is silently made on mis-scaled input.
    """
    from .evaluate import resolve_checkpoint_path
    device = device or get_device(cfg.training.device)
    ckpt_path = resolve_checkpoint_path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg, name_override=ckpt.get("model_name", "unet")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm_stats = ckpt.get("normalize_stats", {"mean": 0.5, "std": 0.25})
    return model, norm_stats, device


def predict_single_image(model, norm_stats, device, image_gray: np.ndarray,
                         image_size: int = 256, threshold: float = 0.5,
                         normalization: str = "zscore_dataset",
                         use_clahe: bool = False, clahe_clip_limit: float = 2.0,
                         clahe_tile_grid: int = 8):
    """
    Run inference on a single grayscale image (numpy array, any input size).

    The preprocessing applied here mirrors the validation-time path in
    LungSegmentationDataset exactly (resize -> optional CLAHE -> normalize),
    so a prediction from the app matches a prediction from the evaluation
    script for the same file.

    Returns:
        prob_map: float32 array in [0,1], image_size x image_size
        pred_mask: uint8 binary mask (0/1), image_size x image_size
        resized_input: the resized uint8 grayscale actually fed to the model
    """
    resized = resize_image(image_gray, image_size)
    model_input = apply_clahe(resized, clahe_clip_limit, clahe_tile_grid) if use_clahe else resized
    norm = normalize_image(model_input, mean=norm_stats.get("mean"),
                           std=norm_stats.get("std"), strategy=normalization)
    tensor = torch.from_numpy(np.ascontiguousarray(norm)).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    pred_mask = (probs >= threshold).astype(np.uint8)
    return probs, pred_mask, resized


def predict_from_config(model, norm_stats, device, image_gray, cfg, threshold=None):
    """Convenience wrapper that pulls all preprocessing options from config."""
    dc = cfg.data
    return predict_single_image(
        model, norm_stats, device, image_gray,
        image_size=dc.image_size,
        threshold=cfg.evaluation.threshold if threshold is None else threshold,
        normalization=dc.get("normalization", "zscore_dataset"),
        use_clahe=dc.get("use_clahe", False),
        clahe_clip_limit=dc.get("clahe_clip_limit", 2.0),
        clahe_tile_grid=dc.get("clahe_tile_grid", 8),
    )


def mask_statistics(pred_mask: np.ndarray, prob_map: np.ndarray = None) -> dict:
    """
    Descriptive statistics for a predicted mask.

    NOTE: areas are reported in PIXELS only. The dataset ships as PNG slices
    with no pixel-spacing (mm/pixel) metadata, so converting to mm^2 or to a
    physical volume would require inventing a scale factor.
    """
    pred_bool = pred_mask.astype(bool)
    n_fg = int(pred_bool.sum())
    total = int(pred_bool.size)
    stats = {
        "height": int(pred_mask.shape[0]),
        "width": int(pred_mask.shape[1]),
        "total_pixels": total,
        "foreground_pixels": n_fg,
        "foreground_percent": (100.0 * n_fg / total) if total else 0.0,
        "bounding_box": None,
        "n_components": 0,
        "mean_foreground_confidence": None,
    }
    if n_fg == 0:
        return stats

    rows = np.where(pred_bool.any(axis=1))[0]
    cols = np.where(pred_bool.any(axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    stats["bounding_box"] = {
        "x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1,
        "width": x1 - x0 + 1, "height": y1 - y0 + 1,
    }

    try:
        import cv2
        n_labels, _ = cv2.connectedComponents(pred_mask.astype(np.uint8))
        stats["n_components"] = int(n_labels - 1)  # label 0 is background
    except Exception:
        stats["n_components"] = None

    if prob_map is not None:
        stats["mean_foreground_confidence"] = float(prob_map[pred_bool].mean())
    return stats
