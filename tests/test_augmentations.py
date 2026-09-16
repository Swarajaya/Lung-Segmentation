"""
Augmentation correctness.

The property that matters most: a geometric transform applied to the image
must be applied IDENTICALLY to the mask, and photometric transforms must not
touch the mask at all. These are verified empirically over many random draws
rather than assumed from the library's documentation.
"""
import numpy as np
import pytest
import albumentations as A

from src.augmentations import get_train_augmentations, get_val_augmentations
from src.config import load_config


@pytest.fixture
def pair():
    """An image whose bright region is exactly its mask, so misalignment shows up."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[18:44, 20:46] = 255
    mask = (img > 0).astype(np.uint8)
    return img, mask


def test_val_pipeline_is_identity(pair):
    img, mask = pair
    out = get_val_augmentations()(image=img, mask=mask)
    assert np.array_equal(out["image"], img)
    assert np.array_equal(out["mask"], mask)


def test_geometric_transforms_keep_image_and_mask_aligned(pair):
    img, mask = pair
    cfg = load_config()
    transform = get_train_augmentations(cfg)
    worst_iou = 1.0
    for _ in range(200):
        out = transform(image=img, mask=mask)
        aug_img_fg = out["image"] > 100
        aug_mask = out["mask"] > 0
        union = (aug_img_fg | aug_mask).sum()
        if union:
            worst_iou = min(worst_iou, (aug_img_fg & aug_mask).sum() / union)
    assert worst_iou > 0.85, f"Image/mask alignment broke (worst IoU {worst_iou:.3f})"


def test_mask_stays_binary_after_augmentation(pair):
    img, mask = pair
    transform = get_train_augmentations(load_config())
    for _ in range(200):
        out = transform(image=img, mask=mask)
        assert set(np.unique(out["mask"]).tolist()).issubset({0, 1})


def test_photometric_transform_never_touches_the_mask(pair):
    """Brightness/contrast/noise applied alone must leave the mask byte-identical."""
    img, mask = pair
    photometric = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
        A.GaussNoise(std_range=(0.05, 0.15), p=1.0),
    ])
    for _ in range(50):
        out = photometric(image=img, mask=mask)
        assert np.array_equal(out["mask"], mask)
        assert not np.array_equal(out["image"], img)


def test_horizontal_flip_mirrors_both_targets(pair):
    img, mask = pair
    out = A.Compose([A.HorizontalFlip(p=1.0)])(image=img, mask=mask)
    assert np.array_equal(out["image"], img[:, ::-1])
    assert np.array_equal(out["mask"], mask[:, ::-1])


def test_vertical_flip_is_disabled_by_default():
    """Axial CT has a fixed anterior/posterior orientation — see src/augmentations.py."""
    assert load_config().augmentation.vertical_flip_prob == 0.0


def test_augmentation_is_configurable_off(pair):
    """Setting every probability to zero must leave the sample essentially unchanged."""
    img, mask = pair
    cfg = load_config()
    cfg.augmentation["horizontal_flip_prob"] = 0.0
    cfg.augmentation["affine_prob"] = 0.0
    cfg.augmentation["brightness_contrast_prob"] = 0.0
    cfg.augmentation["gauss_noise_prob"] = 0.0
    out = get_train_augmentations(cfg)(image=img, mask=mask)
    assert np.array_equal(out["image"], img)
    assert np.array_equal(out["mask"], mask)


def test_augmentation_actually_changes_samples(pair):
    """Guard against a silently no-op pipeline."""
    img, mask = pair
    transform = get_train_augmentations(load_config())
    changed = sum(1 for _ in range(50)
                  if not np.array_equal(transform(image=img, mask=mask)["image"], img))
    assert changed > 25, "Augmentation pipeline barely changes anything"


def test_shapes_are_preserved(pair):
    img, mask = pair
    transform = get_train_augmentations(load_config())
    for _ in range(20):
        out = transform(image=img, mask=mask)
        assert out["image"].shape == img.shape
        assert out["mask"].shape == mask.shape
