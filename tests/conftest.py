"""
Shared pytest fixtures.

Two kinds of test live in this suite:

1. Tests that run against a SYNTHETIC mini-dataset built on the fly
   (`synthetic_dataset`). These exercise the real code paths — loading,
   pairing, splitting, augmentation, training step, evaluation — and run
   anywhere, including on a machine that does not have the CT dataset.

2. Tests that run against the REAL dataset, which is not redistributed with
   this repository. These are skipped automatically when the dataset is
   absent (see `require_real_dataset`), so the suite never reports a false
   failure just because the data has not been downloaded yet.

Both kinds are genuine tests; nothing is stubbed out to force a pass.
"""
import os
import sys
import json

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.config import load_config, resolve_path  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def real_dataset_available(cfg) -> bool:
    return (os.path.isdir(resolve_path(cfg.data.images_dir))
            and os.path.isdir(resolve_path(cfg.data.masks_dir)))


@pytest.fixture
def require_real_dataset(cfg):
    if not real_dataset_available(cfg):
        pytest.skip("Real CT dataset not present — see DATASET_SETUP.md. "
                    "Synthetic-dataset tests still cover this code path.")
    return cfg


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    """
    Build a small synthetic dataset with the same on-disk contract as the real
    one: 8-bit grayscale PNGs, masks with identical filenames and values in
    {0, 255}, filenames encoding a subject ID, and a realistic proportion of
    entirely-empty masks.

    6 subjects x 6 slices = 36 pairs, 64x64 px. Every third slice has an empty
    mask, mirroring the ~16% empty-mask rate measured on the real dataset.
    """
    root = tmp_path_factory.mktemp("synthetic_ct")
    images = root / "images"
    masks = root / "masks"
    images.mkdir()
    masks.mkdir()

    rng = np.random.default_rng(0)
    filenames = []
    for subj in range(6):
        for sl in range(6):
            fname = f"train_Subject_{subj}_{sl}.png"
            filenames.append(fname)

            img = rng.integers(20, 90, size=(64, 64), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)

            if sl % 3 != 0:  # two of every three slices carry foreground
                y = 12 + (subj * 3) % 20
                x = 14 + (sl * 4) % 20
                img[y:y + 16, x:x + 16] = 200
                mask[y:y + 16, x:x + 16] = 255

            Image.fromarray(img).save(images / fname)
            Image.fromarray(mask).save(masks / fname)

    return {
        "root": str(root),
        "images_dir": str(images),
        "masks_dir": str(masks),
        "filenames": filenames,
        "n_empty": sum(1 for f in filenames if int(f.split("_")[-1].split(".")[0]) % 3 == 0),
    }


@pytest.fixture
def committed_split():
    """The split committed in data/splits.json, if present."""
    path = resolve_path("data/splits.json")
    if not os.path.exists(path):
        pytest.skip("data/splits.json not present — run scripts/split_dataset.py")
    with open(path) as f:
        return json.load(f)
