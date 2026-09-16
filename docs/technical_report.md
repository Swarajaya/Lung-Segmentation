# Lung CT Image Segmentation with U-Net — Technical Report

## Abstract

We present a complete, reproducible pipeline for binary semantic segmentation
of axial lung CT slices using a U-Net convolutional encoder–decoder. The
supplied dataset comprises 1,930 paired 256×256 grayscale slices and binary
masks drawn from 61 subjects, with severe class imbalance (foreground = 0.453%
of pixels, background:foreground = 219.7:1) and 15.80% entirely empty masks.
We enforce a subject-level train/validation/test partition to eliminate
patient-level data leakage, verified by automated assertions. Two controlled
experiments were trained on the complete training split (1,352 images): a
BCE-only U-Net baseline without augmentation, and a U-Net with medically
constrained augmentation and a combined BCE+Dice loss. On the held-out test
split of 282 images (236 foreground-positive, 46 empty), the augmented BCE+Dice
model achieved a Dice of 0.2351 on foreground-positive slices versus 0.0517 for
the baseline — a 4.5× improvement. Error analysis shows the baseline collapsed
toward the majority class, missing 192 of 236 positive slices while emitting
only 41 false-positive pixels across the entire test set; the BCE+Dice model
reduced missed slices to 65 at the cost of false alarms on 27 of 46 empty
slices. All training was performed on a single CPU core at reduced capacity
(128×128 input, base_channels=16, 10 epochs); the reported figures are
therefore a genuine lower bound rather than the architecture's achievable
performance. No result in this report was estimated, extrapolated or
fabricated.

## 1. Introduction

Semantic segmentation assigns a class label to every pixel. In medical imaging
this is the operation that yields measurable quantities — lesion extent,
organ boundaries, region areas — which classification (one label per image) and
detection (one box per object) cannot provide. Lung CT segmentation is a
canonical instance: the clinically relevant structures are small, irregular,
and defined by their exact boundary.

This project builds the pipeline end-to-end and, equally importantly,
*evaluates it critically*. The emphasis throughout is on correctness of
methodology — leakage prevention, honest treatment of degenerate cases, and
reporting of failures — rather than on maximising a single headline number.

## 2. Problem definition

**Input:** a single-channel axial lung CT slice, `H×W` uint8.
**Output:** a binary mask of identical spatial dimensions, where 1 marks the
annotated foreground region.
**Model output:** raw logits; a sigmoid and a threshold (default 0.5) convert
these to a binary mask.
**Objective:** maximise overlap with the ground-truth mask while controlling
both false positives and false negatives, under severe class imbalance.

## 3. Dataset

Measured by `scripts/inspect_dataset.py` over every file
(`outputs/metrics/dataset_inspection_report.md`):

| Property | Value |
|---|---|
| Image/mask pairs | 1,930 (0 missing, 0 mismatched, 0 corrupt) |
| Image / mask size, dtype | 256×256, `uint8`, PNG mode `L` |
| Mask pixel values | `{0, 255}` only; 0 non-binary masks |
| Unique subjects | 61 |
| Slices per subject | min 11, max 84, mean 31.6 |
| Empty masks | 305 (15.80%) |
| Foreground fraction | mean 0.453%, median 0.219%, max 3.59% |
| Background : foreground | 219.7 : 1 |
| Image intensity | mean 46.01, std 12.06 |
| Duplicate image groups | 0 |
| Duplicate mask groups | 3 (310 files) |

The three duplicate *mask* groups are expected and benign: all 305 empty masks
are byte-identical to one another. Duplicate *images* — the condition that
would actually indicate a corrupted or double-counted dataset — number zero.

Provenance of this particular file collection could not be verified from the
files alone. This report therefore asserts no specific public source. Genuinely
public alternatives for extending the work include the LIDC-IDRI collection
(TCIA), the Medical Segmentation Decathlon Task06 (Lung), and LUNA16.

## 4. Data analysis

Two properties dominate every downstream design decision.

**Class imbalance.** With a 219.7:1 background-to-foreground ratio, pixel
accuracy is a useless metric and binary cross-entropy — which averages
uniformly over pixels — receives ~99.55% of its gradient signal from
background. Section 8 shows this is not a theoretical concern: the BCE-only
baseline did collapse.

**Empty masks.** 15.80% of slices have no foreground. Dice is 0/0 on such a
slice. Any pipeline must decide explicitly what that means; silently dropping
the cases would hide the model's false-alarm behaviour, and scoring them
uniformly with the rest inflates the headline mean. We do both things
deliberately (Sections 7 and 9).

**Subject structure.** 1,930 slices from 61 subjects means ~32 correlated
slices per patient — the direct motivation for Section 5.

## 5. Leakage prevention

Filenames follow `<origtag>_Subject_<id>_<slice>.png`. Splitting is performed
over **subjects**, never over images:

| Split | Subjects | Images |
|---|---|---|
| train | 43 | 1,352 |
| val | 9 | 296 |
| test | 9 | 282 (236 positive, 46 empty) |

`src/splitting.py::verify_no_leakage` asserts three independent properties: no
subject in two splits, no file in two splits, and every file's parsed subject
belongs to its own split's subject list (which catches a hand-edited or
legacy split file). These assertions run as part of the test suite, so a leaked
split fails CI rather than quietly inflating metrics.

The `train`/`val` tags embedded in the original filenames were **not** trusted:
they define no test set, and we cannot verify they were constructed
subject-wise. Regenerating the split from the seed reproduces the committed
`data/splits.json` byte-for-byte.

## 6. Methodology

### 6.1 Preprocessing

Deterministic and strictly asymmetric between images and masks:

| Step | Image | Mask |
|---|---|---|
| Load | grayscale `L` | grayscale `L` |
| Resize | bilinear | **nearest-neighbour** |
| Contrast | optional CLAHE | never |
| Normalize | configurable | **never** |
| Binarize | — | threshold 127 → `{0,1}` |

Nearest-neighbour mask resizing avoids inventing fractional label values at
boundaries. `tests/test_preprocessing.py` demonstrates empirically that
bilinear resizing of a mask produces 11 distinct values where only 2 are valid,
and that the project's path produces exactly 2.

Normalization strategies: `zscore_dataset` (default; mean/std from the
**training split only**), `zscore_image`, `minmax`, `none`. Computing
statistics on train alone is what keeps normalization free of evaluation-set
leakage; the statistics are stored in the checkpoint so inference reproduces
training-time scaling exactly.

CLAHE is available but **off by default**, keeping the baseline the simplest
reproducible configuration.

### 6.2 Augmentation

Geometric transforms are applied to image and mask with identical sampled
parameters in a single call (nearest-neighbour interpolation for the mask);
photometric transforms are image-only. Alignment is verified empirically over
200 random draws rather than assumed from library documentation.

| Transform | Applied to | Setting |
|---|---|---|
| Horizontal flip | image + mask | p = 0.5 |
| Vertical flip | — | **disabled** (p = 0.0) |
| Affine rotate / translate / scale | image + mask | ±10°, ±5%, 0.95–1.05, p = 0.7 |
| Brightness / contrast | image only | ±0.1, p = 0.3 |
| Gaussian noise | image only | std 0.01–0.05, p = 0.2 |

Vertical flip is disabled on anatomical grounds: an axial slice has a fixed
anterior/posterior orientation, so a vertically flipped slice is anatomically
impossible and would never occur at test time. Horizontal flip is retained
because left/right mirroring remains plausible.

### 6.3 Architecture

U-Net (Ronneberger et al., 2015). Encoder: four stages of
`[Conv3×3-BN-ReLU]×2` followed by 2×2 max-pooling, doubling channel width per
stage. Bottleneck at lowest resolution / highest depth. Decoder: four stages of
upsample → concatenate the corresponding encoder feature map (skip connection)
→ `[Conv3×3-BN-ReLU]×2`. Output: 1×1 convolution to 1 channel of raw logits.

Skip connections are the architecturally essential element: pooling discards
the precise spatial detail needed to place a boundary, and the skips re-inject
it at matching resolution. Decoder stages pad for odd input dimensions, so
non-power-of-two inputs work (tested at 130×130).

Configurable: `in_channels`, `out_channels`, `base_channels`,
`bilinear_upsampling`, and `name` ∈ {`unet`, `residual_unet`,
`attention_unet`}. The variants are single-change ablations; `unet` remains the
primary baseline because reproducibility and interpretability outrank
architectural novelty for this work.

### 6.4 Loss

`L = w_bce · BCEWithLogits + w_dice · SoftDice`, default 0.5 / 0.5.

Dice optimizes `1 − 2|P∩G| / (|P|+|G|)`, which depends only on the foreground
and is therefore far less sensitive to imbalance than BCE. Its weaknesses are
noisy early gradients and the 0/0 case; BCE supplies the stable per-pixel
signal that compensates. Empty-mask handling is explicit:

- `smooth` (default): `(2·TP + s) / (|P| + |G| + s)`. Empty GT + empty
  prediction → s/s = 1 → loss 0 (correct). Empty GT + non-empty prediction →
  `s/(|P|+s) < 1`, a penalty that grows with false-positive area.
- `exclude`: the Dice term is computed over non-empty-GT samples only.
  Provided for ablation; not the default, because it removes the only signal
  penalising hallucinated foreground on empty slices.

Under either setting empty-GT samples always contribute to the BCE term and are
never dropped from the batch. A unit test confirms that on a mask with 0.39%
foreground, an all-background prediction yields BCE < 0.1 but Dice > 0.85 —
quantifying exactly why the Dice term is needed.

### 6.5 Training

Configurable optimizer (Adam/AdamW/SGD/RMSprop), scheduler
(ReduceLROnPlateau/cosine/step/none), early stopping on validation Dice,
gradient clipping, automatic CPU/GPU selection, mixed precision on CUDA
(a no-op on CPU so both paths are identical), best and latest checkpointing,
and resume-from-checkpoint. **Model selection uses validation Dice only; the
test split is never consulted during training.**

## 7. Evaluation metrics

Per-sample: Dice, IoU/Jaccard, precision, recall/sensitivity, specificity, F1,
foreground coverage, and the confusion components. Boundary quality: **HD95**
(95th-percentile symmetric surface distance — the percentile rather than the
maximum, so one stray pixel cannot dominate) and **ASSD**.

Both surface metrics are reported in **pixels**. The dataset carries no
pixel-spacing metadata, so any millimetre figure would be invented. When either
mask is empty the distance is genuinely undefined; those cases are excluded
from the surface average and **the number of contributing cases is reported**,
rather than substituting 0 (which would flatter the model) or the image
diagonal (which would punish it arbitrarily).

Because 15.80% of masks are empty, every metric is reported in three views —
`all`, `positive_only`, `empty_gt_only` — as defined in the README. Section 8
shows why this matters in practice.

## 8. Experimental setup and results

### 8.1 Setup

| Item | Value |
|---|---|
| Config | `config/cpu_real.yaml` |
| Input resolution | 128×128 (dataset native: 256×256) |
| `base_channels` | 16 (target: 32) |
| Parameters | 838,497 |
| Epochs | 10 (target: 60) |
| Batch size | 8 |
| Optimizer / LR | Adam, 1e-4, weight decay 1e-5 |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Hardware | 1 CPU core, no CUDA; ~72 s/epoch |
| Train / val / test | 1,352 / 296 / 282 images (43 / 9 / 9 subjects) |

**Capacity caveat.** Resolution, width and epoch count are all reduced relative
to `config.yaml` because no GPU was available. The results below are a real but
**lower-bound** measurement and must not be read as the architecture's ceiling.

### 8.2 Results

Best validation Dice: 0.1906 (A) and 0.2962 (B).

Test split, 282 images:

| Experiment | Aug | Loss | Dice (all) | **Dice (GT+)** | IoU (GT+) | Precision | Recall | Specificity | Dice (empty GT) | HD95 px | ASSD px |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A `baseline_unet_no_aug` | no | BCE | 0.1957 | **0.0517** | 0.0333 | 0.9787 | 0.1910 | 1.0000 | 0.9348 | 7.32 (n=44) | 4.52 |
| B `unet_aug_bce_dice` | yes | BCE+Dice | 0.2641 | **0.2351** | 0.1614 | 0.5098 | 0.3931 | 0.9973 | 0.4130 | 47.90 (n=171) | 23.53 |

Experiment B improves foreground-positive Dice by **4.5×** over A. The
`Dice (all)` column illustrates the reporting hazard directly: A's 0.1957 is
composed almost entirely of free 1.0 scores on empty slices (its
`Dice (empty GT)` is 0.9348) while its actual segmentation quality is 0.0517.
A single-number report would have badly misrepresented model A.

### 8.3 Smoke-test results (kept separate)

An earlier 2-epoch, 64-image pipeline check at 256×256 produced Dice 0.0079 and
0.0026 (`outputs/metrics/results_table_smoke_test.md`). These verify the
pipeline executes end-to-end and are **not** a measure of quality; they are
stored in separate files so they cannot be confused with §8.2.

### 8.4 Not executed

Full training at target settings (256×256, `base_channels=32`, 60 epochs) was
infeasible: ~0.73 s/image/epoch on one CPU core implies roughly 16 hours per
experiment. **No number for that configuration appears anywhere in this
repository.** Experiment C (`attention_unet_aug_bce_dice`) is implemented and
test-covered but was not run on real data, and so appears in no results table.

## 9. Visual results

`outputs/figures/` contains, all from real test-split predictions: a stratified
gallery (success / average / difficult / failure / empty-GT, selected by
quartiles of the measured Dice distribution so worst cases are always
included); 6-panel figures per tier showing original, ground truth, prediction,
overlay, error map and probability map; dedicated success/failure/error-map
figures; and training curves including loss, Dice, IoU, learning rate and a
train-vs-validation generalization-gap panel.

## 10. Error analysis

Automated, over 282 test images (236 positive, 46 empty). Categories are not
mutually exclusive.

| Category | A | B |
|---|---|---|
| `missed_positive` | 192 | 65 |
| `low_dice` (<0.5) | 228 | 184 |
| `low_iou` (<0.35) | 229 | 188 |
| `correct_empty` | 43/46 | 19/46 |
| `false_alarm_on_empty` | 3 | 27 |
| Total FP pixels | 41 | 12,219 |
| Total FN pixels | 15,370 | 9,359 |

**Interpretation.** Model A exhibits textbook majority-class collapse: near-
perfect precision (0.9787) and specificity (1.0000) with recall 0.1910, and 41
false-positive pixels in the entire test set. It learned to predict almost
nothing. This is precisely the failure the Dice term is designed to prevent,
and observing it directly is the strongest empirical justification for the loss
design in §6.4.

Model B shifts the operating point: missed slices fall from 192 to 65, but
false alarms on empty slices rise from 3 to 27 and total FP pixels rise to
12,219. Its much larger HD95 (47.90 px vs 7.32) reflects scattered
false-positive components far from the true boundary rather than a poorly
placed single contour — note also that HD95 is defined for 171 of B's cases
versus only 44 of A's, because A so often predicts nothing at all.

Clinically, a missed lesion is generally more costly than a false alarm, so B's
trade is the more defensible one; its false-positive rate is nonetheless the
obvious next target. A recurring, visually identifiable failure mode is a
false-positive region on the left chest wall, outside the lung field entirely —
removable by a lung-field mask or a connected-component area filter.

## 11. Limitations

1. Results are capacity-limited (128×128, `base_channels=16`, 10 epochs) and
   are a lower bound.
2. The best model false-alarms on 27 of 46 empty test slices.
3. 2D slice-wise segmentation discards inter-slice volumetric context.
4. Pixel areas cannot be converted to physical units — no pixel spacing exists.
5. 61 subjects, 9 in test: per-subject variance is high.
6. Unverified provenance, so results are not comparable to published benchmarks.
7. Single split, no cross-validation.
8. Threshold fixed at 0.5 rather than tuned on validation.
9. No clinical validation of any kind.

## 12. Future work

Full-resolution GPU training; morphological/lung-field post-processing to
remove chest-wall false positives; validation-tuned decision threshold; proper
attention- and residual-U-Net ablations; subject-level k-fold cross-validation;
2.5D inputs for volumetric context; Tversky or focal-Tversky loss for direct
FP/FN control.

## 13. Conclusion

We built and validated a complete lung CT segmentation pipeline with verified
leakage prevention, explicit handling of the dataset's 15.80% empty masks, and
three-view metric reporting that separates genuine segmentation quality from
the free scores that empty slices confer. Two controlled experiments trained on
the full training split show that medically constrained augmentation combined
with a BCE+Dice loss raises foreground-positive Dice from 0.0517 to 0.2351,
while error analysis documents exactly what each model does wrong — the
baseline's majority-class collapse and the improved model's false-alarm
trade-off. All figures reported are measured; the compute-limited nature of the
training runs is stated explicitly, and no result for an unexecuted
configuration is claimed anywhere.

## 14. References

- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks
  for Biomedical Image Segmentation.* MICCAI.
- Oktay, O. et al. (2018). *Attention U-Net: Learning Where to Look for the
  Pancreas.* MIDL.
- Milletari, F., Navab, N., Ahmadi, S.-A. (2016). *V-Net: Fully Convolutional
  Neural Networks for Volumetric Medical Image Segmentation.* 3DV. (Dice loss.)
- Taghanaki, S. A. et al. (2021). *Deep semantic segmentation of natural and
  medical images: a review.* Artificial Intelligence Review.
- Public lung-CT datasets for extension: LIDC-IDRI (TCIA); Medical Segmentation
  Decathlon Task06 Lung; LUNA16.
