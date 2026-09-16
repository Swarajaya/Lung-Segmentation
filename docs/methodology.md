# Methodology Summary

A condensed, presentation-friendly walk-through of the pipeline. For full
detail and justification see `docs/technical_report.md`.

```
Raw dataset (images/, masks/)
        │
        ▼
scripts/inspect_dataset.py  ──► outputs/metrics/dataset_inspection_report.{json,md}
        │
        ▼
scripts/split_dataset.py    ──► data/splits.json  (subject-level train/val/test)
        │
        ▼
src/dataset.py + src/preprocessing.py + src/augmentations.py
   (load → resize → binarize mask → normalize → augment [train only])
        │
        ▼
src/models.py  (U-Net)  ──trained with──►  src/losses.py (BCE+Dice)
        │
        ▼
src/train.py  (train/val loop, checkpointing, early stopping)
        │
        ▼
outputs/checkpoints/<experiment>_best.pt
        │
        ├──► src/evaluate.py + scripts/evaluate_model.py  ──► outputs/metrics/results_table_*.{csv,md}
        │
        ├──► src/predict.py + scripts/generate_results.py ──► outputs/figures/*.png
        │
        └──► app/app.py (Streamlit demo)
```

## Key methodological decisions and why

| Decision | Why |
|---|---|
| Subject-level (not image-level) split | Multiple slices share a subject ID; random image splitting leaks patient identity across train/eval |
| Nearest-neighbor resize for masks | Preserves exact label values; bilinear would invent fractional boundary values |
| Threshold=127 mask binarization | Measured mask values are exactly `{0,255}` — no ambiguity |
| BCE + Dice combined loss | Measured ~0.45% foreground fraction → BCE alone gives a weak signal; Dice alone is less stable early in training |
| Empty-mask edge case → Dice/IoU = 1.0 | 15.8% of masks are empty; treating this as NaN would corrupt aggregate metrics |
| Smoke test vs. full training modes | Detected hardware (1 CPU core, no GPU) cannot complete full training; smoke test proves pipeline correctness without pretending otherwise |
