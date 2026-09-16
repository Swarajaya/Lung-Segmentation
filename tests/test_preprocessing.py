"""Preprocessing: determinism, mask integrity, normalization strategies, CLAHE."""
import numpy as np
import pytest

from src.preprocessing import (
    resize_image, resize_mask, binarize_mask, normalize_image, apply_clahe,
    preprocess_image, preprocess_mask, validate_pair, VALID_NORMALIZATIONS,
)


@pytest.fixture
def img():
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, size=(80, 80), dtype=np.uint8)


@pytest.fixture
def mask():
    m = np.zeros((80, 80), dtype=np.uint8)
    m[20:50, 25:55] = 255
    return m


def test_preprocessing_is_deterministic(img):
    a = preprocess_image(img, 64, use_clahe=True, mean=0.4, std=0.2)
    b = preprocess_image(img, 64, use_clahe=True, mean=0.4, std=0.2)
    assert np.array_equal(a, b)


def test_mask_resize_keeps_only_original_values(mask):
    """Nearest-neighbour resizing must never invent intermediate label values."""
    out = resize_mask(mask, 32)
    assert set(np.unique(out).tolist()).issubset({0, 255})


def test_mask_bilinear_would_blend_but_we_do_not_use_it(mask):
    """Guard test: proves the nearest/bilinear distinction actually matters here."""
    import cv2
    # 80 -> 121 is a non-integer ratio, so bilinear interpolation must produce
    # intermediate values at the rectangle's edges.
    blended = cv2.resize(mask, (121, 121), interpolation=cv2.INTER_LINEAR)
    assert not set(np.unique(blended).tolist()).issubset({0, 255}), (
        "Test fixture is too simple to detect blending")
    # Nearest-neighbour, used by the project, does not.
    assert set(np.unique(resize_mask(mask, 121)).tolist()).issubset({0, 255})


def test_binarize_mask_is_strictly_binary(mask):
    b = binarize_mask(mask, 127)
    assert set(np.unique(b).tolist()).issubset({0, 1})
    assert b.dtype == np.uint8


def test_preprocess_mask_never_normalizes(mask):
    out = preprocess_mask(mask, 32, 127)
    assert set(np.unique(out).tolist()).issubset({0, 1})
    assert out.dtype == np.uint8
    assert out.shape == (32, 32)


def test_all_normalization_strategies_produce_finite_output(img):
    for s in VALID_NORMALIZATIONS:
        out = normalize_image(img, mean=0.4, std=0.2, strategy=s)
        assert np.isfinite(out).all(), s
        assert out.dtype == np.float32


def test_unknown_normalization_strategy_raises(img):
    with pytest.raises(ValueError):
        normalize_image(img, strategy="definitely_not_a_strategy")


def test_zscore_image_gives_zero_mean_unit_std(img):
    out = normalize_image(img, strategy="zscore_image")
    assert abs(float(out.mean())) < 1e-4
    assert abs(float(out.std()) - 1.0) < 1e-3


def test_minmax_maps_into_unit_range(img):
    out = normalize_image(img, strategy="minmax")
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_normalization_handles_constant_image():
    flat = np.full((16, 16), 77, dtype=np.uint8)
    for s in ("minmax", "zscore_image"):
        out = normalize_image(flat, strategy=s)
        assert np.isfinite(out).all()


def test_zscore_dataset_falls_back_without_stats(img):
    out = normalize_image(img, mean=None, std=None, strategy="zscore_dataset")
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_clahe_preserves_shape_and_dtype(img):
    out = apply_clahe(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_resize_image_shape(img):
    assert resize_image(img, 128).shape == (128, 128)


def test_validate_pair_accepts_valid(img, mask):
    assert validate_pair(img, mask) == []


def test_validate_pair_detects_shape_mismatch(img):
    issues = validate_pair(img, np.zeros((10, 10), dtype=np.uint8))
    assert any("shape_mismatch" in i for i in issues)


def test_validate_pair_detects_bad_mask_values(img):
    bad = np.full((80, 80), 137, dtype=np.uint8)
    issues = validate_pair(img, bad)
    assert any("unexpected_mask_values" in i for i in issues)
