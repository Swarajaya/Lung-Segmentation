# Results table (smoke_test)

All numbers below were computed by `scripts/evaluate.py` on the held-out test split. `dice_all` includes empty-ground-truth slices (which score 1.0 when correctly predicted empty); `dice_positive_only` covers only slices that actually contain foreground and is the figure that measures segmentation quality. HD95/ASSD are in pixels and are averaged over the cases where they are defined.

| Experiment | Model | Aug | Loss | Dice (all) | Dice (GT+) | IoU (GT+) | Precision | Recall | Specificity | HD95 px | ASSD px | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_unet_no_aug | unet | False | bce | 0.0065 | 0.0094 | 0.0047 | 0.0033 | 0.4146 | 0.9648 | 165.589 | 90.824 | SMOKE TEST (tiny data subset, 2 epochs) — proves the pipeline runs end-to-end. NOT a measure of model quality. |
| unet_aug_bce_dice | unet | True | bce_dice | 0.007 | 0.0101 | 0.0051 | 0.0042 | 0.3396 | 0.9905 | 155.218 | 83.599 | SMOKE TEST (tiny data subset, 2 epochs) — proves the pipeline runs end-to-end. NOT a measure of model quality. |
