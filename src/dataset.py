"""PyTorch Dataset for paired lung CT images and segmentation masks."""
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    load_image_grayscale, load_mask, resize_image, resize_mask,
    binarize_mask, normalize_image, apply_clahe, validate_pair,
)

# Filenames in the supplied dataset follow `<origtag>_Subject_<id>_<slice>.png`.
SUBJECT_RE = re.compile(r'Subject_(\d+)_', re.IGNORECASE)


def subject_id_from_filename(fname: str):
    """Extract the subject/patient ID from a filename, or None if not encoded."""
    m = SUBJECT_RE.search(fname)
    return m.group(1) if m else None


class LungSegmentationDataset(Dataset):
    """
    Loads (image, mask) pairs from disk given a list of filenames shared
    between the images/ and masks/ directories.

    Robustness: every sample is validated on load (shape match, no NaN/Inf,
    mask values within the expected set). Invalid samples raise a clear
    error rather than silently producing garbage tensors.

    Order of operations per sample:
        load -> validate -> resize (bilinear img / nearest mask) -> binarize
        mask -> optional CLAHE (image only) -> augment (shared geometry)
        -> normalize (image only) -> to tensor

    CLAHE is applied before augmentation so that the photometric augmentations
    perturb the same enhanced signal the model sees at inference time.
    """

    def __init__(self, filenames, images_dir, masks_dir, image_size=256,
                 mask_threshold=127, transform=None, normalize_stats=None,
                 max_samples=None, normalization="zscore_dataset",
                 use_clahe=False, clahe_clip_limit=2.0, clahe_tile_grid=8):
        self.filenames = list(filenames)
        if max_samples is not None:
            self.filenames = self.filenames[:max_samples]
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.transform = transform
        self.normalize_stats = normalize_stats  # (mean, std) or None
        self.normalization = normalization
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid

    def __len__(self):
        return len(self.filenames)

    @classmethod
    def from_config(cls, cfg, filenames, images_dir, masks_dir, transform=None,
                    normalize_stats=None, max_samples=None):
        """Build a dataset using the data.* section of config.yaml."""
        dc = cfg.data
        return cls(
            filenames, images_dir, masks_dir,
            image_size=dc.image_size,
            mask_threshold=dc.mask_threshold,
            transform=transform,
            normalize_stats=normalize_stats,
            max_samples=max_samples,
            normalization=dc.get("normalization", "zscore_dataset"),
            use_clahe=dc.get("use_clahe", False),
            clahe_clip_limit=dc.get("clahe_clip_limit", 2.0),
            clahe_tile_grid=dc.get("clahe_tile_grid", 8),
        )

    def subject_ids(self):
        """Subject IDs of every sample in this dataset (None where unparsable)."""
        return [subject_id_from_filename(f) for f in self.filenames]

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.images_dir, fname)
        mask_path = os.path.join(self.masks_dir, fname)

        img = load_image_grayscale(img_path)
        mask = load_mask(mask_path)

        issues = validate_pair(img, mask)
        if issues:
            raise ValueError(f"Invalid sample '{fname}': {issues}")

        img = resize_image(img, self.image_size)
        mask = resize_mask(mask, self.image_size)
        mask_bin = binarize_mask(mask, self.mask_threshold)

        if self.use_clahe:
            img = apply_clahe(img, self.clahe_clip_limit, self.clahe_tile_grid)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask_bin)
            img = augmented["image"]
            mask_bin = augmented["mask"]

        mean, std = (self.normalize_stats if self.normalize_stats else (None, None))
        img_norm = normalize_image(img, mean=mean, std=std, strategy=self.normalization)

        # Mask stays a label field: cast only, never normalized.
        mask_bin = np.asarray(mask_bin).astype(np.uint8)

        img_tensor = torch.from_numpy(np.ascontiguousarray(img_norm)).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_bin)).unsqueeze(0).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "filename": fname,
            "subject_id": subject_id_from_filename(fname) or "",
        }


def compute_dataset_stats(filenames, images_dir, image_size=256, max_samples=500,
                          use_clahe=False, clahe_clip_limit=2.0, clahe_tile_grid=8):
    """
    Compute mean/std pixel intensity (in [0,1]) over a sample of the TRAINING
    split only. Computing these on train alone (never on val/test) is what
    keeps the normalization step free of evaluation-set leakage.

    If CLAHE is enabled it is applied here too, so the statistics describe the
    same signal the model will actually receive.
    """
    sample = list(filenames)[:max_samples]
    if not sample:
        return 0.5, 0.25
    total, total_sq, n = 0.0, 0.0, 0
    for fname in sample:
        img = load_image_grayscale(os.path.join(images_dir, fname))
        img = resize_image(img, image_size)
        if use_clahe:
            img = apply_clahe(img, clahe_clip_limit, clahe_tile_grid)
        arr = img.astype(np.float64) / 255.0
        total += arr.sum()
        total_sq += (arr ** 2).sum()
        n += arr.size
    mean = total / n
    var = max(total_sq / n - mean ** 2, 0.0)
    return float(mean), float(np.sqrt(var))
