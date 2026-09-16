# Dataset Inspection Report

This report was generated automatically by `scripts/inspect_dataset.py` by reading every image/mask file on disk. No value below is assumed.

- Number of image files: **1930**
- Number of mask files: **1930**
- Number of valid image/mask pairs: **1930**
- Missing masks for images: 0
- Missing images for masks: 0
- Unique subjects (parsed from filenames): **61**
- Filenames not matching the `Subject_<id>_<slice>` pattern: 0
- Original split tags embedded in filenames: {'train': 1832, 'val': 98}

## Image properties
- Formats: {'PNG': 1930}
- Color modes: {'L': 1930}
- Sizes: {'(256, 256)': 1930}
- Dtypes: {'uint8': 1930}
- Corrupt/unreadable images: 0
- Mean pixel intensity across sample: 46.01 (std 12.06)

## Mask properties
- Formats: {'PNG': 1930}
- Color modes: {'L': 1930}
- Sizes: {'(256, 256)': 1930}
- Corrupt/unreadable masks: 0
- Unique pixel value sets found: {'(0,)': 305, '(0, 255)': 1625}
- Empty masks (no foreground pixel): 305 (15.8% of pairs)
- Masks with more than 2 unique values (non-binary): 0
- Image/mask pairs with mismatched dimensions: 0
- Mean foreground pixel fraction: 0.453% (median 0.219%, max 3.59%)

- Mask dtypes: {'uint8': 1930}

## Class imbalance
- Foreground pixels across the whole dataset: 573,053 of 126,484,480
- Foreground pixel ratio: **0.4531%**
- Background:foreground ratio: **219.7 : 1**

This is severe class imbalance, and it is the reason the project uses a Dice-based loss term rather than BCE alone: a model predicting background everywhere would already score very high pixel accuracy.

## Subjects
- Subjects summarised: 61
- Slices per subject: min 11, max 84, mean 31.6

Subjects contribute unequal numbers of slices, which is exactly why the train/val/test split is made at SUBJECT level (see `scripts/split_dataset.py`).

## Duplicates
- Duplicate IMAGE pixel-content groups: 0 (0 files involved)
- Duplicate MASK pixel-content groups: 3 (310 files involved) — expected to be large, since every empty mask is byte-identical to every other empty mask

## Interpretation
- Masks are strictly binary with values `{0, 255}` -> a threshold of 127 cleanly recovers a `{0,1}` foreground/background mask.
- A non-trivial fraction of masks are entirely empty (no foreground). This is expected for CT slices that do not intersect the annotated structure, and is handled explicitly in the metric code (an empty ground truth with an empty prediction scores a perfect Dice/IoU of 1.0, not undefined/NaN).
- Filenames encode a subject ID, which enables leak-free SUBJECT-LEVEL train/val/test splitting (see `scripts/split_dataset.py`).