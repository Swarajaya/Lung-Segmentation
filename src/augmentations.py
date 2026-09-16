"""
Medically-reasonable data augmentation.

Guarantee enforced here: every GEOMETRIC transform is applied to the image and
its mask with the SAME sampled random parameters, in a single
`transform(image=..., mask=...)` call. Albumentations applies a spatial
transform to the `mask` target using the same parameter draw as the `image`
target, with nearest-neighbour interpolation for the mask, so the mask can
never drift out of alignment with its image. `tests/test_augmentations.py`
asserts this property empirically rather than trusting it.

Every PHOTOMETRIC transform (brightness, contrast, noise) is image-only.
Albumentations does not apply pixel-level transforms to mask targets, which is
exactly what we want: a binary label field has no brightness to adjust, and
adding noise to it would produce invalid non-{0,1} labels.

Choices specific to axial lung CT:
- Horizontal flip is allowed: mirroring left/right lung remains an
  anatomically plausible image.
- Vertical flip is DISABLED by default (config: vertical_flip_prob: 0.0)
  because an axial slice has a fixed anterior/posterior orientation and
  flipping it produces an anatomically impossible image that the model would
  never see at test time.
- Rotation/scale/translation are kept small (±10°, ±5%, ±5%) to simulate
  realistic patient positioning variation rather than implausible poses.
- Brightness/contrast jitter is mild, simulating scanner/windowing variation.
- Gaussian noise is mild, simulating acquisition noise at low dose.
"""
import albumentations as A


def get_train_augmentations(cfg, seed: int = None) -> A.Compose:
    """
    Build the training augmentation pipeline from config.augmentation.

    The Compose is explicitly seeded. Albumentations >=2.0 gives each Compose
    its own random generator seeded from system entropy, so WITHOUT this the
    augmented pipeline would not be reproducible even after torch/numpy seeds
    are fixed — two runs of the same config would diverge. Passing the project
    seed makes an augmented training run repeatable end to end.
    """
    ac = cfg.augmentation
    seed = cfg.seed if seed is None else seed
    transforms = []

    # ---- geometric (applied identically to image AND mask) ----
    if ac.horizontal_flip_prob > 0:
        transforms.append(A.HorizontalFlip(p=ac.horizontal_flip_prob))
    if ac.get("vertical_flip_prob", 0.0) > 0:
        transforms.append(A.VerticalFlip(p=ac.vertical_flip_prob))

    transforms.append(
        A.Affine(
            rotate=(-ac.rotation_degrees, ac.rotation_degrees),
            translate_percent={
                "x": (-ac.translate_fraction, ac.translate_fraction),
                "y": (-ac.translate_fraction, ac.translate_fraction),
            },
            scale=tuple(ac.scale_range),
            interpolation=1,       # bilinear for the image
            mask_interpolation=0,  # nearest-neighbor for the mask (no label blending)
            p=ac.get("affine_prob", 0.7),
        )
    )

    # ---- photometric (image only) ----
    if ac.brightness_contrast_prob > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=ac.brightness_limit,
                contrast_limit=ac.contrast_limit,
                p=ac.brightness_contrast_prob,
            )
        )
    if ac.get("gauss_noise_prob", 0) > 0:
        lo, hi = ac.get("gauss_noise_std_range", [0.01, 0.05])
        transforms.append(
            A.GaussNoise(std_range=(float(lo), float(hi)), p=ac.gauss_noise_prob)
        )

    return _compose(transforms, seed)


def get_val_augmentations() -> A.Compose:
    """No augmentation at validation/test time — identity pipeline."""
    return _compose([], 0)


def _compose(transforms, seed):
    """Compose with an explicit seed where the installed albumentations supports it."""
    try:
        return A.Compose(transforms, seed=seed)
    except TypeError:
        # albumentations < 2.0 has no `seed` argument; it draws from the global
        # numpy RNG instead, which src.utils.set_seed already fixes.
        return A.Compose(transforms)
