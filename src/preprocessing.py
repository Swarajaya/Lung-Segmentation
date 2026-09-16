"""
Preprocessing utilities: reading, resizing and normalizing images/masks.

Design decisions (documented, not assumed):
- Images are read as single-channel grayscale (the supplied dataset is 8-bit
  grayscale lung CT slices, confirmed by dataset inspection).
- Images are resized with bilinear interpolation (smooth intensity signal).
- Masks are resized with NEAREST-NEIGHBOR interpolation to avoid inventing
  fractional/blended label values at object boundaries (a standard practice
  for segmentation masks; see docs/viva_questions.md).
- Masks are binarized using a fixed threshold on the raw pixel value,
  established from the actual inspected mask value set {0, 255}.
- Masks are NEVER intensity-normalized, CLAHE-enhanced, or bilinearly
  resampled. A mask is a label field, not a signal: applying image
  normalization to it would destroy the {0,1} semantics. The only operations
  applied to masks are nearest-neighbor resize, binarization, and geometric
  augmentation shared with the image.

Normalization strategies (config: data.normalization)
-----------------------------------------------------
  "zscore_dataset" (default) - scale to [0,1], then standardize with the
        mean/std computed from the TRAINING split only. Training-set-only
        statistics avoid leaking val/test intensity information.
  "zscore_image"   - scale to [0,1], then standardize per-image using that
        image's own mean/std. Useful when scanner/protocol intensity drift
        is large; makes each slice self-normalizing.
  "minmax"         - scale to [0,1] using the image's own min/max.
  "none"           - scale to [0,1] by dividing by 255 only.

All strategies are deterministic: the same input file always produces the
same tensor (no randomness anywhere in this module).
"""
import numpy as np
import cv2
from PIL import Image

VALID_NORMALIZATIONS = ("zscore_dataset", "zscore_image", "minmax", "none")


def load_image_grayscale(path: str) -> np.ndarray:
    """Load an image file as a single-channel uint8 numpy array."""
    img = Image.open(path).convert("L")
    return np.array(img)


def load_mask(path: str) -> np.ndarray:
    """Load a mask file as a single-channel uint8 numpy array."""
    m = Image.open(path).convert("L")
    return np.array(m)


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize an image with bilinear interpolation."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Resize a mask with nearest-neighbor interpolation (preserves label integrity)."""
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def binarize_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Convert a raw {0,255}-valued mask into a strict {0,1} binary mask."""
    return (mask >= threshold).astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization on a uint8 grayscale image.

    CLAHE equalizes contrast in local tiles rather than globally, which on CT
    slices makes low-contrast soft-tissue / lesion boundaries more visible
    without blowing out the already-bright bone and already-dark air regions
    the way plain global histogram equalization does. `clip_limit` bounds the
    local contrast amplification so noise in homogeneous regions (e.g. the
    air outside the patient) is not amplified into spurious texture.

    Deterministic; applied to IMAGES ONLY, never to masks.
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_grid), int(tile_grid)))
    return clahe.apply(img)


def normalize_image(img: np.ndarray, mean: float = None, std: float = None,
                    strategy: str = "zscore_dataset") -> np.ndarray:
    """
    Scale a uint8 image to float32 and normalize it per the chosen strategy.

    `mean`/`std` are the dataset (training-split) statistics, used only by the
    "zscore_dataset" strategy. If they are not supplied, "zscore_dataset"
    degrades gracefully to a plain [0,1] scaling rather than silently
    inventing statistics.
    """
    if strategy not in VALID_NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization strategy '{strategy}'. "
            f"Valid options: {VALID_NORMALIZATIONS}"
        )

    img = img.astype(np.float32) / 255.0

    if strategy == "none":
        return img

    if strategy == "minmax":
        lo, hi = float(img.min()), float(img.max())
        if hi - lo < 1e-8:
            return np.zeros_like(img)
        return (img - lo) / (hi - lo)

    if strategy == "zscore_image":
        m, s = float(img.mean()), float(img.std())
        if s < 1e-8:
            return np.zeros_like(img)
        return (img - m) / s

    # zscore_dataset
    if mean is not None and std is not None and std > 1e-8:
        return (img - mean) / std
    return img


def preprocess_image(img: np.ndarray, size: int, use_clahe: bool = False,
                     clahe_clip_limit: float = 2.0, clahe_tile_grid: int = 8,
                     mean: float = None, std: float = None,
                     strategy: str = "zscore_dataset") -> np.ndarray:
    """
    Full deterministic image preprocessing: resize -> optional CLAHE -> normalize.

    CLAHE is applied AFTER resizing and BEFORE normalization, because it is
    defined on uint8 intensity values and its tile grid should correspond to
    the resolution the model actually sees.
    """
    out = resize_image(img, size)
    if use_clahe:
        out = apply_clahe(out, clahe_clip_limit, clahe_tile_grid)
    return normalize_image(out, mean=mean, std=std, strategy=strategy)


def preprocess_mask(mask: np.ndarray, size: int, threshold: int = 127) -> np.ndarray:
    """Full mask preprocessing: nearest-neighbor resize -> binarize. No normalization."""
    return binarize_mask(resize_mask(mask, size), threshold)


def validate_pair(img: np.ndarray, mask: np.ndarray, allowed_mask_values=(0, 255)) -> list:
    """Return a list of issue strings for a given image/mask pair (empty = OK)."""
    issues = []
    if img.shape != mask.shape:
        issues.append(f"shape_mismatch: image {img.shape} vs mask {mask.shape}")
    if img.size == 0:
        issues.append("image_empty")
    if not np.isfinite(img.astype(np.float64)).all():
        issues.append("image_has_nan_or_inf")
    if not np.isfinite(mask.astype(np.float64)).all():
        issues.append("mask_has_nan_or_inf")
    uniq = set(np.unique(mask).tolist())
    if not uniq.issubset(set(allowed_mask_values)):
        issues.append(f"unexpected_mask_values: {sorted(uniq)}")
    return issues
