# Dataset Setup

The raw CT dataset is **not** included in this repository. This file explains
exactly where to put it and how to verify the placement was correct.

## 1. Required layout

Place the dataset so that the project root contains:

```
lung-segmentation/
└── data/
    └── lung-cancer-vision-v1/
        ├── images/   # 1,930 grayscale lung CT slice PNGs
        └── masks/    # 1,930 binary segmentation mask PNGs
```

**A mask must have exactly the same filename as its image.** That filename is
the only thing pairing them — see `src/dataset.py`. If your archive extracts
straight to `data/images` and `data/masks`, move them one level down:

```bash
mkdir -p data/lung-cancer-vision-v1
mv data/images data/masks data/lung-cancer-vision-v1/
```

If you keep the data somewhere else, point the project at it instead of moving
files — edit `config.yaml`:

```yaml
data:
  images_dir: "/absolute/or/relative/path/to/images"
  masks_dir:  "/absolute/or/relative/path/to/masks"
```

Nothing in `src/` or `scripts/` hard-codes a dataset path; every path comes
from `config.yaml`.

## 2. Expected filename format

```
<origtag>_Subject_<subject_id>_<slice_id>.png
```

for example `train_Subject_17_287.png`. The `Subject_<id>` part is **required**
for leak-free splitting — `scripts/split_dataset.py` parses it to guarantee no
patient appears in two splits. If your filenames do not encode a subject ID,
the split script will stop with a clear error rather than silently producing a
leaky image-level split.

The `train`/`val` prefix is an artefact of how the files were distributed. The
project deliberately ignores it and derives its own split; see
"Why subject-level splitting" in the README.

## 3. Verify the placement

```bash
python scripts/inspect_dataset.py    # reads every file, writes a measured report
python scripts/split_dataset.py      # builds + leakage-checks the split
python -m pytest tests/ -q           # 129 tests, none skipped when data is present
```

`inspect_dataset.py` should report **0** corrupt files, **0** missing pairs and
**0** dimension mismatches. If the dataset is absent, the dataset-dependent
tests skip automatically instead of failing, and the rest of the suite still
runs.

## 4. Properties of the supplied dataset (all measured)

Every figure below was produced by `scripts/inspect_dataset.py` reading the
actual files — see `outputs/metrics/dataset_inspection_report.md`.

| Property | Measured value |
|---|---|
| Image/mask pairs | 1,930 (0 missing, 0 mismatched) |
| Image size / dtype / mode | 256×256, `uint8`, 8-bit grayscale PNG |
| Mask size / dtype | 256×256, `uint8` |
| Mask pixel values | strictly `{0, 255}` — 0 non-binary masks |
| Unique subjects | 61 |
| Slices per subject | min 11, max 84, mean 31.6 |
| Empty masks | 305 (15.80%) |
| Mean foreground fraction | 0.453% (median 0.219%, max 3.59%) |
| Background : foreground | 219.7 : 1 |
| Corrupt files | 0 images, 0 masks |
| Duplicate image groups | 0 |
| Duplicate mask groups | 3 (310 files) — expected: all 305 empty masks are byte-identical |

The severe class imbalance in the last rows is the direct justification for the
Dice-based loss term (see `src/losses.py`), and the 305 empty masks are why
evaluation reports separate "all images" and "foreground-positive only" figures
(see `src/metrics.py`).

## 5. Provenance

The original public source of this specific file collection could not be
verified from the files alone, so this project makes **no** claim about its
provenance (no specific Kaggle competition or TCIA collection is asserted).
`docs/technical_report.md` lists genuinely public lung-CT segmentation datasets
you can cite or use to extend the work.

## 6. Keeping the dataset out of version control

`.gitignore` already excludes `data/lung-cancer-vision-v1/`. Do not commit the
raw imaging data.
