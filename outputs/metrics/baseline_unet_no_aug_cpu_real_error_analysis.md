# Error Analysis — `baseline_unet_no_aug` (cpu_real)

Generated automatically by `scripts/error_analysis.py` from the actual per-sample evaluation output. Every number below is measured.

- Images analysed: **282**
- With non-empty ground truth: **236**
- With empty ground truth: **46**

## Failure categories

Categories are **not mutually exclusive** — one image can be both `low_dice` and `high_false_positive`.

| Category | Count |
|---|---|
| `correct_empty` | 43 |
| `false_alarm_on_empty` | 3 |
| `low_dice` | 228 |
| `low_iou` | 229 |
| `missed_positive` | 192 |

Thresholds used: {'low_dice_threshold': 0.5, 'low_iou_threshold': 0.35, 'high_fp_pixel_threshold': 500, 'high_fn_pixel_threshold': 500}

## False positive vs false negative balance

- Total false-positive pixels: **41**
- Total false-negative pixels: **15,370**
- FP/FN ratio: **0.003**

A ratio far above 1 means the model over-segments (false alarms); far below 1 means it under-segments (missed structure). The two errors are not clinically equivalent, so they are reported separately rather than collapsed into one figure.

## Worst cases by Dice (non-empty ground truth only)

| File | Subject | Dice | IoU | TP | FP | FN |
|---|---|---|---|---|---|---|
| `train_Subject_14_174.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 3 |
| `train_Subject_14_175.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 13 |
| `train_Subject_14_176.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 18 |
| `train_Subject_14_177.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 19 |
| `train_Subject_14_178.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 21 |
| `train_Subject_14_179.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 20 |
| `train_Subject_14_180.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 23 |
| `train_Subject_14_181.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 30 |
| `train_Subject_14_182.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 35 |
| `train_Subject_14_183.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 43 |
| `train_Subject_14_184.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 29 |
| `train_Subject_14_185.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 24 |
| `train_Subject_14_186.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 14 |
| `train_Subject_14_187.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 12 |
| `train_Subject_14_188.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 10 |
| `train_Subject_14_189.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 7 |
| `train_Subject_14_190.png` | 14 | 0.0000 | 0.0000 | 0 | 0 | 6 |
| `train_Subject_15_76.png` | 15 | 0.0000 | 0.0000 | 0 | 0 | 2 |
| `train_Subject_15_77.png` | 15 | 0.0000 | 0.0000 | 0 | 0 | 4 |
| `train_Subject_15_78.png` | 15 | 0.0000 | 0.0000 | 0 | 0 | 7 |

## Per-subject mean Dice (worst 10, non-empty slices only)

If a few subjects dominate the bottom of this table, the model is failing on particular patients (anatomy/protocol) rather than uniformly across slices.

| Subject | Positive slices | Mean Dice |
|---|---|---|
| 14 | 17 | 0.0000 |
| 15 | 15 | 0.0000 |
| 1 | 21 | 0.0000 |
| 47 | 14 | 0.0000 |
| 8 | 70 | 0.0000 |
| 57 | 18 | 0.0000 |
| 7 | 7 | 0.0247 |
| 17 | 45 | 0.0996 |
| 40 | 29 | 0.2601 |
