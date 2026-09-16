# Error Analysis — `unet_aug_bce_dice` (cpu_real)

Generated automatically by `scripts/error_analysis.py` from the actual per-sample evaluation output. Every number below is measured.

- Images analysed: **282**
- With non-empty ground truth: **236**
- With empty ground truth: **46**

## Failure categories

Categories are **not mutually exclusive** — one image can be both `low_dice` and `high_false_positive`.

| Category | Count |
|---|---|
| `correct_empty` | 19 |
| `false_alarm_on_empty` | 27 |
| `low_dice` | 184 |
| `low_iou` | 188 |
| `missed_positive` | 65 |

Thresholds used: {'low_dice_threshold': 0.5, 'low_iou_threshold': 0.35, 'high_fp_pixel_threshold': 500, 'high_fn_pixel_threshold': 500}

## False positive vs false negative balance

- Total false-positive pixels: **12,219**
- Total false-negative pixels: **9,359**
- FP/FN ratio: **1.306**

A ratio far above 1 means the model over-segments (false alarms); far below 1 means it under-segments (missed structure). The two errors are not clinically equivalent, so they are reported separately rather than collapsed into one figure.

## Worst cases by Dice (non-empty ground truth only)

| File | Subject | Dice | IoU | TP | FP | FN |
|---|---|---|---|---|---|---|
| `train_Subject_14_174.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 3 |
| `train_Subject_14_175.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 13 |
| `train_Subject_14_176.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 18 |
| `train_Subject_14_177.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 19 |
| `train_Subject_14_178.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 21 |
| `train_Subject_14_186.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 14 |
| `train_Subject_14_187.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 12 |
| `train_Subject_14_188.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 10 |
| `train_Subject_14_189.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 7 |
| `train_Subject_14_190.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 6 |
| `train_Subject_15_76.png` | 15 | 0.0000 | 0.0000 | 0 | 21 | 2 |
| `train_Subject_15_77.png` | 15 | 0.0000 | 0.0000 | 0 | 20 | 4 |
| `train_Subject_15_78.png` | 15 | 0.0000 | 0.0000 | 0 | 17 | 7 |
| `train_Subject_15_79.png` | 15 | 0.0000 | 0.0000 | 0 | 19 | 10 |
| `train_Subject_15_80.png` | 15 | 0.0000 | 0.0000 | 0 | 86 | 15 |
| `train_Subject_15_81.png` | 15 | 0.0000 | 0.0000 | 0 | 84 | 17 |
| `train_Subject_15_89.png` | 15 | 0.0000 | 0.0000 | 0 | 13 | 2 |
| `train_Subject_15_90.png` | 15 | 0.0000 | 0.0000 | 0 | 34 | 2 |
| `train_Subject_17_280.png` | 17 | 0.0000 | 0.0000 | 0 | 73 | 2 |
| `train_Subject_17_313.png` | 17 | 0.0000 | 0.0000 | 0 | 15 | 33 |

## Per-subject mean Dice (worst 10, non-empty slices only)

If a few subjects dominate the bottom of this table, the model is failing on particular patients (anatomy/protocol) rather than uniformly across slices.

| Subject | Positive slices | Mean Dice |
|---|---|---|
| 1 | 21 | 0.0000 |
| 47 | 14 | 0.0192 |
| 57 | 18 | 0.0593 |
| 15 | 15 | 0.1373 |
| 14 | 17 | 0.1779 |
| 40 | 29 | 0.2909 |
| 8 | 70 | 0.2980 |
| 7 | 7 | 0.3453 |
| 17 | 45 | 0.3854 |
