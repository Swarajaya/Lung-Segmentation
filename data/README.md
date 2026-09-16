# Data directory

Place the dataset here as:

```
data/
└── lung-cancer-vision-v1/
    ├── images/   # grayscale lung CT slice PNGs
    └── masks/    # corresponding binary segmentation mask PNGs (same filenames as images/)
```

Each mask file must have the **same filename** as its corresponding image file
(this is how the pipeline pairs them — see `src/dataset.py`).

## Actual properties of the supplied dataset (measured, see `outputs/metrics/dataset_inspection_report.md`)

- 1,930 image/mask pairs, all matched (no missing pairs)
- All files: 256×256, 8-bit grayscale PNG
- Masks are strictly binary-valued `{0, 255}`
- Filenames encode a subject ID: `<origtag>_Subject_<subject_id>_<slice_id>.png`,
  used for leak-free subject-level splitting
- 61 unique subjects
- ~15.8% of masks are entirely empty (slices with no annotated foreground)
- No corrupted files, no duplicate images, no dimension mismatches detected

We could not verify the original public source of this specific file
collection from the files alone; the README/report therefore does **not**
claim a specific provenance (e.g. a specific Kaggle competition or TCIA
collection) for the supplied files. See `docs/technical_report.md` §20
(References) for genuinely public lung-CT segmentation datasets you can cite
or use to extend this project.

## After placing the data

Run, in order:

```bash
python scripts/inspect_dataset.py     # regenerates the inspection report from your files
python scripts/split_dataset.py       # regenerates the subject-level train/val/test split
```
