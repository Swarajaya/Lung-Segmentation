# Lung CT Image Segmentation with U-Net

Binary semantic segmentation of lesion/structure regions in axial lung CT
slices, built as a complete, reproducible computer-vision pipeline: dataset
validation → leak-free subject-level splitting → preprocessing → augmentation
→ U-Net → BCE/Dice losses → training → multi-metric evaluation → error
analysis → visual results → interactive demo.

> **Medical disclaimer.** This is a research/educational prototype and is not
> intended for clinical diagnosis or medical decision-making. No claim of
> clinical accuracy is made, and the model has not been validated on any
> clinical population.

---

## 1. Problem statement

Given a single-channel axial lung CT slice, predict a **per-pixel** binary mask
marking the annotated foreground region. This is *segmentation*, not
classification or detection: the output has the same spatial resolution as the
input, and every pixel receives its own label.

The task is hard for three measured reasons, all confirmed by dataset
inspection rather than assumed:

1. **Extreme class imbalance** — foreground occupies **0.453%** of pixels on
   average; the background-to-foreground ratio is **219.7 : 1**. A model that
   predicts "background everywhere" already achieves ~99.5% pixel accuracy.
2. **Many empty slices** — **15.80%** of masks contain no foreground at all,
   so naive averaging of Dice hands the model free perfect scores.
3. **Patient-correlated data** — 1,930 slices come from only **61 subjects**,
   so a random image-level split would leak patient anatomy between train and
   test.

## 2. Objective

Produce accurate lung/lesion segmentation masks while demonstrating the full
CV pipeline *correctly*: no data leakage, no fabricated metrics, no silently
dropped edge cases, and an evaluation that reports the model's failures as
prominently as its successes.

## 3. Dataset

All figures below were measured by `scripts/inspect_dataset.py` reading every
file on disk. See `outputs/metrics/dataset_inspection_report.md`.

| Property | Measured value |
|---|---|
| Image/mask pairs | 1,930 (0 missing, 0 mismatched) |
| Image size / dtype | 256×256, `uint8`, 8-bit grayscale PNG |
| Mask size / dtype | 256×256, `uint8` |
| Mask pixel values | strictly `{0, 255}`; 0 non-binary masks |
| Unique subjects | 61 |
| Slices per subject | min 11, max 84, mean 31.6 |
| Empty masks | 305 (15.80%) |
| Mean foreground fraction | 0.453% (median 0.219%, max 3.59%) |
| Background : foreground | 219.7 : 1 |
| Mean image intensity | 46.01 (std 12.06) |
| Corrupt files | 0 images, 0 masks |
| Duplicate image groups | 0 |
| Duplicate mask groups | 3 (310 files) — expected, all 305 empty masks are byte-identical |

The dataset is **not** redistributed with this repository. See
[`DATASET_SETUP.md`](DATASET_SETUP.md) for placement instructions.

Provenance could not be verified from the files alone, so this project makes no
claim about the original public source.

## 4. Why subject-level splitting

Filenames encode a subject ID (`<origtag>_Subject_<id>_<slice>.png`). Adjacent
slices of one patient share anatomy, scanner and reconstruction. A random
*image-level* split would place slices from the same patient in both train and
test, letting the model score well by recognising anatomy it memorised — the
metric would measure memorisation, not generalisation to a **new patient**,
which is the only thing that matters clinically.

Every slice from a given subject therefore goes into exactly one split:

| Split | Subjects | Images | Empty masks |
|---|---|---|---|
| train | 43 | 1,352 | see `outputs/metrics/split_report.md` |
| val | 9 | 296 | " |
| test | 9 | 282 (236 with foreground, 46 empty) | " |

`src/splitting.py::verify_no_leakage` runs three independent checks (no shared
subject, no shared file, and every file's subject belongs to its own split) and
is enforced as an automated test, so a leaked split **fails the test suite**
rather than silently inflating results.

The original filenames also carry a `train`/`val` tag from however the files
were distributed. We deliberately ignore it: it defines no test set and we
cannot verify it was built subject-wise.

## 5. Pipeline

```
inspect_dataset.py → split_dataset.py → train.py → evaluate.py
                                                 → error_analysis.py
                                                 → generate_results.py → app/app.py
```

### Preprocessing (`src/preprocessing.py`)
Deterministic, configurable, and strictly separated for images and masks.

- Images: grayscale load → bilinear resize → *optional* CLAHE → normalize.
- Masks: **nearest-neighbour** resize → threshold at 127 → `{0,1}`.
- Masks are **never** normalized, CLAHE-enhanced or bilinearly resampled.
  A mask is a label field, not a signal. `tests/test_preprocessing.py` proves
  bilinear resizing would invent intermediate label values and that the
  project's nearest-neighbour path does not.
- Normalization strategies (`data.normalization`): `zscore_dataset` (default,
  statistics from the **training split only** — no val/test leakage),
  `zscore_image`, `minmax`, `none`.

### Augmentation (`src/augmentations.py`)
- Geometric (shared identically by image **and** mask): horizontal flip,
  small affine rotation ±10°, translation ±5%, scale 0.95–1.05.
- Photometric (**image only**): brightness/contrast jitter, mild Gaussian noise.
- **Vertical flip is disabled by default** — an axial slice has a fixed
  anterior/posterior orientation, so flipping it produces an anatomically
  impossible image the model would never see at test time.
- Alignment is verified empirically over 200 random draws, not assumed.

### Model (`src/models.py`)
U-Net (Ronneberger et al., 2015) is the **primary baseline**: encoder
(Conv-BN-ReLU ×2 + maxpool, doubling channels), bottleneck, decoder (upsample →
concatenate skip → Conv-BN-ReLU ×2), 1×1 output conv emitting **raw logits**.
Skip connections restore the spatial precision that pooling discards, which is
exactly what pixel-accurate boundaries need.

Two optional single-change variants are available (`model.name`):
`residual_unet` (residual blocks) and `attention_unet` (additive attention
gates on each skip, motivated by the 219:1 imbalance). U-Net remains the
baseline — reproducibility and interpretability beat architecture novelty.

### Losses (`src/losses.py`)
`bce`, `dice`, `bce_dice` (weighted, default 0.5/0.5).

Dice directly optimizes region overlap, so unlike BCE it does not let the
99.55% background pixels dominate the gradient. **Empty masks are handled
explicitly**: with `empty_handling: smooth`, an empty GT with an empty
prediction scores loss ≈ 0 (correct), while an empty GT with a non-empty
prediction incurs a loss that grows with the false-positive area. Empty-GT
samples are never dropped from a batch — they always contribute to the BCE term.

### Evaluation (`src/metrics.py`, `src/evaluate.py`)
Dice, IoU, precision, recall/sensitivity, specificity, F1, foreground coverage,
plus **HD95** and **ASSD** boundary distances (in **pixels** — the dataset
carries no mm/pixel metadata, so millimetre figures would be invented).

Because 16% of masks are empty, every metric is reported in **three views**:

- **all** — the whole test set. Empty-GT/empty-prediction pairs score 1.0 here.
- **positive_only** — only slices whose ground truth contains foreground. *This
  is the number that measures segmentation quality.*
- **empty_gt_only** — only empty slices; measures false-alarm behaviour.

Nothing is silently excluded; empty cases are moved into their own reported
subgroup. Surface metrics are undefined when either mask is empty, so those
cases are excluded from the surface average **and the count of contributing
cases is reported** rather than substituting 0 or the image diagonal.

---

## 6. Results

**Read this section carefully — three different kinds of result are reported
and they are not interchangeable.**

### 6a. Real-data training run (`cpu_real`) — ACTUALLY EXECUTED

Trained on the **complete training split** (all 1,352 images from 43 subjects),
validated on all 296 validation images, evaluated on all 282 test images.

Config: [`config/cpu_real.yaml`](config/cpu_real.yaml) — **128×128** input,
`base_channels=16`, 10 epochs, Adam (lr 1e-4), ReduceLROnPlateau, CPU only
(single core, ~72 s/epoch, 838,497 parameters).

> **This is a capacity-reduced run.** It uses lower resolution, a narrower
> network and 10 epochs instead of the 256×256 / `base_channels=32` / 60-epoch
> settings in `config.yaml`. The numbers below are a genuine but **lower
> bound** — they are *not* the architecture's achievable performance. No GPU
> was available in this environment.

Best validation Dice: **0.1906** (baseline) and **0.2962** (augmented).

Test-set results (`outputs/metrics/results_table_cpu_real.md`):

| Experiment | Aug | Loss | Dice (all) | **Dice (GT+)** | IoU (GT+) | Precision | Recall | Specificity | Dice (empty GT) | HD95 px | ASSD px |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_unet_no_aug` | no | BCE | 0.1957 | **0.0517** | 0.0333 | 0.9787 | 0.1910 | 1.0000 | 0.9348 | 7.32 (n=44) | 4.52 |
| `unet_aug_bce_dice` | yes | BCE+Dice | 0.2641 | **0.2351** | 0.1614 | 0.5098 | 0.3931 | 0.9973 | 0.4130 | 47.90 (n=171) | 23.53 |

**Experiment B (augmentation + BCE+Dice) outperforms the baseline by 4.5× on
the metric that matters (Dice on foreground-positive slices: 0.2351 vs
0.0517).** This is a measured result, and it is exactly the behaviour the loss
design predicts.

Note also how misleading the `Dice (all)` column is on its own: the baseline's
0.1957 is inflated almost entirely by free 1.0 scores on empty slices
(`Dice (empty GT)` = 0.9348). Its real segmentation quality is 0.0517. This is
precisely why the three-view reporting exists.

### 6b. Error analysis — ACTUALLY EXECUTED

From `outputs/metrics/*_cpu_real_error_analysis.md`, over 282 test images
(236 foreground-positive, 46 empty). Categories are **not** mutually exclusive.

| Category | `baseline_unet_no_aug` | `unet_aug_bce_dice` |
|---|---|---|
| `missed_positive` (GT has foreground, prediction empty) | **192** | **65** |
| `low_dice` (<0.5, positive GT only) | 228 | 184 |
| `low_iou` (<0.35, positive GT only) | 229 | 188 |
| `correct_empty` | 43 / 46 | 19 / 46 |
| `false_alarm_on_empty` | 3 | 27 |
| Total FP pixels | 41 | 12,219 |
| Total FN pixels | 15,370 | 9,359 |

This is the project's central finding and it is fully interpretable:

- The **BCE-only baseline collapsed toward the majority class**. Precision
  0.9787 with recall 0.1910 and only 41 false-positive pixels in the entire
  test set means it predicts almost nothing, and is right when it does. It
  missed 192 of 236 positive slices entirely. This is the exact failure mode
  the Dice loss term exists to prevent, observed directly.
- The **BCE+Dice model trades precision for recall** — misses drop from 192 to
  65, but it now false-alarms on 27 of 46 empty slices. Its much larger HD95
  (47.9 px) reflects scattered false-positive components far from the true
  boundary.
- In a screening context a false negative (missed lesion) is more dangerous
  than a false alarm, so B's trade is the more clinically sensible one — but
  its false-positive rate is clearly the thing to fix next.

A recurring, visible failure mode in the gallery: a false-positive blob on the
**left chest wall** outside the lung field, which a post-processing lung-field
mask or a connected-component area filter would remove cheaply.

### 6c. Smoke-test results (preserved, separate) — ACTUALLY EXECUTED

`outputs/metrics/results_table_smoke_test.md`, from an earlier run at
256×256 / `base_channels=32` on **64 training images for 2 epochs**. Dice
0.0079 and 0.0026. These prove the pipeline runs end-to-end; they are **not** a
measure of model quality and are kept in separate files so they can never be
confused with §6a.

### 6d. NOT executed

- **Full training at target settings** (256×256, `base_channels=32`, 60
  epochs). Infeasible here: ~0.73 s/image/epoch on one CPU core ⇒ roughly 16
  hours per experiment. No GPU was available. **No number for this
  configuration is reported anywhere in this repository.**
- **Experiment C** (`attention_unet_aug_bce_dice`) is defined in `config.yaml`
  with `enabled: false`. Its forward/backward path is covered by tests and it
  trains correctly, but it was **not** run on the real dataset, so it appears
  in no results table.

### 6e. Visual results

In `outputs/figures/` (all from real test-split predictions):

- `unet_aug_bce_dice_cpu_real_gallery.png` — stratified gallery: success /
  average / difficult / failure / empty-GT cases, chosen by **quartiles of the
  actual Dice distribution** so the model's worst cases are always shown. This
  is not a highlight reel.
- `*_cpu_real_{success,average,difficult,failure,empty_gt}_panel.png` — 6-panel
  figures: original CT, ground truth, prediction, overlay, error map, and
  probability map.
- `cpu_real_success_case.png`, `cpu_real_failure_case.png`,
  `cpu_real_failure_case_error_map.png`, `cpu_real_probability_map_example.png`
- `*_cpu_real_training_curves.png` — loss, Dice, IoU, learning rate, validation
  Dice/IoU, and a train-vs-val generalization-gap panel.

---

## 7. Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+. CPU-only works; CUDA is used automatically when present.
Then place the dataset as described in [`DATASET_SETUP.md`](DATASET_SETUP.md).

## 8. Usage

Every command below was executed in producing this repository.

```bash
# 1. Validate the dataset (writes a measured report)
python scripts/inspect_dataset.py

# 2. Build + leakage-check the subject-level split
python scripts/split_dataset.py

# 3a. Fast pipeline check (~2 min CPU)
python scripts/train.py --mode smoke_test

# 3b. The real CPU run reported in §6a (~25 min total on one core)
python scripts/train.py --mode full --config config/cpu_real.yaml --label cpu_real

# 4. Evaluate on the held-out test split
python scripts/evaluate.py --mode full --config config/cpu_real.yaml --label cpu_real

# 5. Automated error analysis
python scripts/error_analysis.py --mode full --config config/cpu_real.yaml --label cpu_real

# 6. Visual gallery and case figures
python scripts/generate_results.py --mode full --config config/cpu_real.yaml \
    --label cpu_real --experiment unet_aug_bce_dice --n-eval 140

# 7. Inference on images or a directory
python scripts/predict.py --input data/lung-cancer-vision-v1/images \
    --config config/cpu_real.yaml --experiment unet_aug_bce_dice \
    --limit 6 --save-figure

# 8. Interactive demo
streamlit run app/app.py

# 9. Tests
python -m pytest tests/ -q
```

`--label` keeps each run's outputs in separate files, so a new run can never
overwrite previously validated results.

### Full training on a GPU

The settings in `config.yaml` (256×256, `base_channels=32`, 60 epochs, batch 16,
mixed precision, early stopping patience 10) are intended for a GPU:

```bash
python scripts/train.py --mode full --label gpu_full
python scripts/evaluate.py --mode full --label gpu_full
python scripts/error_analysis.py --mode full --label gpu_full
python scripts/generate_results.py --mode full --label gpu_full --experiment unet_aug_bce_dice
```

Add `--include-optional` to also run the attention-U-Net experiment. Mixed
precision activates automatically on CUDA and is a no-op on CPU. Expect roughly
8–16 GB VRAM at batch 16 and 256×256; halve the batch size if memory-limited.

Training resumes from a checkpoint:

```bash
python scripts/train.py --mode full --experiment unet_aug_bce_dice \
    --resume outputs/checkpoints/unet_aug_bce_dice_last.pt
```

(The `cpu_real` run in §6a was itself produced with `--resume` across several
sessions, so this path is exercised, not just implemented.)

## 9. Project structure

```
├── README.md, DATASET_SETUP.md, requirements.txt, config.yaml
├── config/cpu_real.yaml        # CPU-feasible real-data run settings
├── data/                       # dataset goes here (not redistributed)
│   ├── README.md, splits.json
├── src/
│   ├── config.py preprocessing.py dataset.py splitting.py augmentations.py
│   ├── models.py losses.py metrics.py train.py evaluate.py
│   ├── error_analysis.py predict.py visualization.py utils.py
├── scripts/
│   ├── inspect_dataset.py split_dataset.py train.py evaluate.py
│   ├── predict.py error_analysis.py generate_results.py
├── app/app.py                  # Streamlit demo
├── tests/                      # 129 tests
├── outputs/{figures,metrics,checkpoints,predictions}/
└── docs/{methodology,technical_report,viva_questions,presentation_notes}.md
```

`scripts/train_model.py` and `scripts/evaluate_model.py` remain as thin aliases
for the older command names.

## 10. Streamlit demo

Upload a CT slice (or pick a dataset sample) and get: original, predicted mask,
overlay, probability map, and statistics — foreground pixels, foreground
percentage, image dimensions, connected components, mean confidence, and the
bounding box of the predicted region. A threshold slider shows the
precision/recall trade-off live. When a ground-truth mask exists for the chosen
file, Dice/IoU/precision/recall/HD95 and an error map are shown too. The
predicted mask can be downloaded as a PNG.

Normalization statistics are read **from the checkpoint**, so the app
preprocesses exactly as the model was trained.

## 11. Reproducibility

- Single seed (`config.yaml: seed`) fixes Python, NumPy and PyTorch RNGs, and
  is passed explicitly to the albumentations `Compose` — without that, an
  augmented run is **not** reproducible even with torch/numpy seeded. This was
  found by a test, not by inspection.
- The subject split is a deterministic function of the seed; regenerating it
  reproduces the committed `data/splits.json` exactly.
- Normalization statistics are computed from the training split only and stored
  in the checkpoint.
- No path is hard-coded; everything comes from a config file.
- `torch.backends.cudnn.deterministic = True`, `benchmark = False`.
- A reproducibility test asserts two identical runs produce identical loss
  histories.

## 12. Limitations

- Results in §6a are **capacity-limited** (128×128, 10 epochs) and are a lower
  bound, not the architecture's ceiling.
- The best model still false-alarms on 27 of 46 empty test slices.
- 2D slice-wise segmentation ignores 3D volumetric context between slices.
- Pixel areas cannot be converted to physical area/volume — no pixel-spacing
  metadata exists in this dataset.
- 61 subjects is small; 9 test subjects means per-subject variance is high and
  a single difficult patient noticeably moves the mean.
- Dataset provenance is unverified, so results are not comparable to any
  published benchmark.
- Single train/val/test split — no cross-validation.
- No clinical validation whatsoever.

## 13. Future improvements

1. Train at full resolution on a GPU (the single highest-impact change).
2. Post-process with a lung-field mask or connected-component area filter to
   remove the chest-wall false positives seen in the gallery.
3. Tune the decision threshold on the **validation** split rather than fixing
   it at 0.5 — the precision/recall balance in §6a suggests real headroom.
4. Run the attention-U-Net and residual-U-Net ablations properly.
5. Subject-level k-fold cross-validation for tighter confidence intervals.
6. 2.5D input (adjacent slices as channels) to recover volumetric context.
7. Tversky/focal-Tversky loss to control the FP/FN balance directly.

## 14. Medical disclaimer

This is a research/educational prototype and is not intended for clinical
diagnosis or medical decision-making.
