"""Dataset loading, image/mask pairing, and per-sample validation."""
import os

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import LungSegmentationDataset, compute_dataset_stats, subject_id_from_filename
from src.config import resolve_path


def build_ds(sd, **kw):
    return LungSegmentationDataset(sd["filenames"], sd["images_dir"], sd["masks_dir"],
                                   image_size=64, mask_threshold=127, **kw)


def test_subject_id_extraction():
    assert subject_id_from_filename("train_Subject_17_287.png") == "17"
    assert subject_id_from_filename("nothing.png") is None


def test_dataset_length_and_sample_shapes(synthetic_dataset, cfg):
    ds = build_ds(synthetic_dataset)
    assert len(ds) == len(synthetic_dataset["filenames"])
    s = ds[0]
    assert s["image"].shape == (1, 64, 64)
    assert s["mask"].shape == (1, 64, 64)
    assert s["image"].dtype == torch.float32 and s["mask"].dtype == torch.float32


def test_mask_tensor_is_strictly_binary(synthetic_dataset):
    ds = build_ds(synthetic_dataset)
    for i in range(len(ds)):
        uniq = set(ds[i]["mask"].unique().tolist())
        assert uniq.issubset({0.0, 1.0}), uniq


def test_image_and_mask_are_paired_by_filename(synthetic_dataset):
    """A sample's mask must come from the file with the SAME name as its image."""
    ds = build_ds(synthetic_dataset)
    for i in range(len(ds)):
        s = ds[i]
        expected = np.array(Image.open(
            os.path.join(synthetic_dataset["masks_dir"], s["filename"])).convert("L"))
        expected_bin = (expected >= 127).astype(np.uint8)
        assert np.array_equal(s["mask"][0].numpy().astype(np.uint8), expected_bin)


def test_filename_and_subject_are_returned(synthetic_dataset):
    s = build_ds(synthetic_dataset)[0]
    assert s["filename"] in synthetic_dataset["filenames"]
    assert s["subject_id"].isdigit()


def test_max_samples_caps_the_dataset(synthetic_dataset):
    assert len(build_ds(synthetic_dataset, max_samples=5)) == 5


def test_empty_masks_are_loaded_not_skipped(synthetic_dataset):
    """Empty masks must be kept — the project handles them, it does not hide them."""
    ds = build_ds(synthetic_dataset)
    n_empty = sum(1 for i in range(len(ds)) if ds[i]["mask"].sum() == 0)
    assert n_empty == synthetic_dataset["n_empty"] > 0


def test_missing_mask_file_raises_clearly(synthetic_dataset):
    ds = LungSegmentationDataset(["definitely_missing.png"], synthetic_dataset["images_dir"],
                                 synthetic_dataset["masks_dir"], image_size=64)
    with pytest.raises((FileNotFoundError, OSError)):
        _ = ds[0]


def test_shape_mismatch_is_detected(synthetic_dataset, tmp_path):
    """A mask of the wrong size must raise, not be silently resized into place."""
    bad_masks = tmp_path / "bad_masks"
    bad_masks.mkdir()
    fname = synthetic_dataset["filenames"][0]
    Image.fromarray(np.zeros((32, 16), dtype=np.uint8)).save(bad_masks / fname)
    ds = LungSegmentationDataset([fname], synthetic_dataset["images_dir"],
                                 str(bad_masks), image_size=64)
    with pytest.raises(ValueError, match="shape_mismatch"):
        _ = ds[0]


def test_non_binary_mask_is_rejected(synthetic_dataset, tmp_path):
    bad_masks = tmp_path / "grey_masks"
    bad_masks.mkdir()
    fname = synthetic_dataset["filenames"][0]
    Image.fromarray(np.full((64, 64), 128, dtype=np.uint8)).save(bad_masks / fname)
    ds = LungSegmentationDataset([fname], synthetic_dataset["images_dir"],
                                 str(bad_masks), image_size=64)
    with pytest.raises(ValueError, match="unexpected_mask_values"):
        _ = ds[0]


def test_dataset_stats_are_in_unit_range(synthetic_dataset):
    mean, std = compute_dataset_stats(synthetic_dataset["filenames"],
                                      synthetic_dataset["images_dir"], image_size=64)
    assert 0.0 <= mean <= 1.0 and 0.0 <= std <= 1.0


def test_dataset_stats_are_deterministic(synthetic_dataset):
    a = compute_dataset_stats(synthetic_dataset["filenames"], synthetic_dataset["images_dir"], 64)
    b = compute_dataset_stats(synthetic_dataset["filenames"], synthetic_dataset["images_dir"], 64)
    assert a == b


def test_clahe_changes_the_image_but_not_the_mask(synthetic_dataset):
    plain = build_ds(synthetic_dataset, use_clahe=False)[1]
    clahe = build_ds(synthetic_dataset, use_clahe=True)[1]
    assert not torch.allclose(plain["image"], clahe["image"])
    assert torch.equal(plain["mask"], clahe["mask"])


def test_dataloader_batches_correctly(synthetic_dataset):
    from torch.utils.data import DataLoader
    loader = DataLoader(build_ds(synthetic_dataset), batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 1, 64, 64)
    assert batch["mask"].shape == (4, 1, 64, 64)
    assert len(batch["filename"]) == 4


# --- real dataset (skipped when the data is not installed) ------------------

def test_real_dataset_loads_and_pairs(require_real_dataset, committed_split):
    cfg = require_real_dataset
    files = committed_split["test_files"][:8]
    ds = LungSegmentationDataset.from_config(
        cfg, files, resolve_path(cfg.data.images_dir), resolve_path(cfg.data.masks_dir))
    assert len(ds) == len(files)
    s = ds[0]
    assert s["image"].shape == (1, cfg.data.image_size, cfg.data.image_size)
    assert set(s["mask"].unique().tolist()).issubset({0.0, 1.0})


def test_real_dataset_every_image_has_a_mask(require_real_dataset):
    cfg = require_real_dataset
    images = set(os.listdir(resolve_path(cfg.data.images_dir)))
    masks = set(os.listdir(resolve_path(cfg.data.masks_dir)))
    assert not (images - masks), f"Images without masks: {sorted(images - masks)[:5]}"
    assert not (masks - images), f"Masks without images: {sorted(masks - images)[:5]}"
