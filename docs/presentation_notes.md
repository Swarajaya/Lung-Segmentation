# Professor Demonstration — 5–10 Minute Flow

Each section has a **say this** line (roughly what to speak) and a **show this**
line (what to have on screen). Total ≈ 8 minutes at a normal pace.

Have these open before you start:
1. Terminal at the project root
2. `outputs/metrics/dataset_inspection_report.md`
3. `outputs/figures/unet_aug_bce_dice_cpu_real_gallery.png`
4. `outputs/metrics/results_table_cpu_real.md`
5. `outputs/metrics/unet_aug_bce_dice_cpu_real_error_analysis.md`
6. A browser tab with `streamlit run app/app.py` already running

---

## 1. Problem (30 s)

**Show:** the gallery image, just the first tile.

**Say:** "The task is binary semantic segmentation of axial lung CT slices —
a label for every pixel, not one label per image. Segmentation is what gives
you measurable extent and boundaries, which classification and detection
can't. I'll show the pipeline, the results, and — importantly — where the
model fails."

## 2. Dataset (45 s)

**Show:** `outputs/metrics/dataset_inspection_report.md`.

**Say:** "1,930 image–mask pairs, 256×256 grayscale, from 61 subjects. Every
number here was measured by a script that reads every file — nothing is
assumed. Two properties drive every design decision: foreground is only
0.45% of pixels, a 220-to-1 imbalance, and 15.8% of masks are completely
empty."

## 3. Dataset validation (45 s)

**Show:** run it live — `python scripts/inspect_dataset.py`.

**Say:** "This checks pairing, missing files, corruption, duplicates,
dimensions, mask dtype, mask value sets, empty masks and class balance. Zero
corrupt files, zero missing pairs, zero dimension mismatches. It reports three
duplicate *mask* groups, which is expected — all 305 empty masks are identical
— but zero duplicate *images*, which is the one that would actually indicate a
problem."

## 4. Leakage prevention (60 s) — *the methodological core, don't rush it*

**Show:** `python scripts/split_dataset.py`, then `outputs/metrics/split_report.md`.

**Say:** "Filenames encode a subject ID. Thirty-two slices come from the same
patient on average, and adjacent slices are nearly identical. If I split
randomly by image, the same patient lands in train and test, and the model
scores well by recognising anatomy it memorised — that measures memorisation,
not generalisation to a new patient, which is the only thing that matters
clinically. So I split by subject: 43 / 9 / 9 subjects, 1,352 / 296 / 282
images. Three independent leakage assertions run, and they're part of the test
suite — a leaked split fails CI, it doesn't quietly inflate my numbers."

*Expected question — "why not use the train/val tags in the filenames?"*
"They define no test set and I can't verify they were built subject-wise, so I
derive my own split and document that."

## 5. Model (45 s)

**Say:** "U-Net — encoder, bottleneck, decoder, skip connections. Pooling in
the encoder throws away the precise spatial detail you need to place a
boundary; the skip connections re-inject it at matching resolution. That's the
one idea that makes U-Net work for pixel-accurate medical segmentation. I also
implemented residual and attention variants, but plain U-Net stays the baseline
— reproducibility beats architecture novelty here."

## 6. Loss and empty masks (45 s)

**Say:** "With 220-to-1 imbalance, plain BCE gets 99.55% of its gradient from
background, so it converges to predicting nothing. Dice optimizes region
overlap, which only depends on the foreground. I combine them. And empty masks
are handled explicitly — the smoothing term means empty ground truth plus
empty prediction gives zero loss, which is correct, while predicting something
on an empty slice is penalised in proportion to the area. I never drop those
cases."

## 7. Training (30 s)

**Show:** `outputs/figures/unet_aug_bce_dice_cpu_real_training_curves.png`.

**Say:** "Trained on the complete training split — all 1,352 images. Loss,
Dice, IoU, learning rate, and a train-versus-validation gap panel. Be upfront:
this ran on one CPU core with no GPU, so I reduced to 128×128, 16 base channels
and 10 epochs. These are real measured numbers but they're a lower bound, not
what the architecture can do."

## 8. Metrics (60 s) — *the strongest point you make*

**Show:** `outputs/metrics/results_table_cpu_real.md`.

**Say:** "Dice, IoU, precision, recall, specificity, plus HD95 and ASSD for
boundary quality, in pixels — the dataset has no pixel spacing, so millimetres
would be invented. Now look at the two Dice columns. The baseline shows 0.1957
overall, but only **0.0517** on slices that actually contain foreground. The
difference is free 1.0 scores from empty slices. That's why I report three
views — all images, foreground-positive only, and empty-ground-truth only. The
augmented BCE+Dice model gets 0.2351 on positive slices, four and a half times
the baseline."

## 9. Error analysis (75 s) — *where you show critical thinking*

**Show:** the error-analysis markdown, then the gallery.

**Say:** "This is the part I find most interesting. The baseline has precision
0.98 and specificity 1.0 — which sounds excellent until you see recall is 0.19
and it produced 41 false-positive pixels in the entire test set. It missed 192
of 236 positive slices. It learned to predict almost nothing. That is exactly
the majority-class collapse the Dice loss exists to prevent, and I observed it
directly rather than just citing it.

The BCE+Dice model cuts misses from 192 to 65, but now false-alarms on 27 of
46 empty slices. In screening, a missed lesion is worse than a false alarm, so
that's the better trade — but the false positives are clearly what to fix next.

And in the gallery you can see the specific failure: a blob on the left chest
wall, outside the lung field entirely. A lung-field mask or a component-size
filter would remove that cheaply."

**Point out:** the gallery is stratified by quartiles of the actual Dice
distribution — success, average, difficult, failure, empty — so the worst cases
are always shown. "This isn't a highlight reel."

## 10. Streamlit demo (60 s)

**Show:** the running app. Upload or pick a sample → Run segmentation.

**Say:** "Original, predicted mask, overlay, and the probability map — the
confidence, not just the thresholded output. Statistics: foreground pixels,
percentage, dimensions, connected components, bounding box. The threshold
slider shows the precision/recall trade-off live — drag it down and watch
recall and false positives both rise. Where ground truth exists it also shows
Dice, IoU and an error map. And the medical disclaimer is always on screen."

**Drag the threshold slider** — it's the most memorable thing in the demo.

## 11. Limitations (30 s)

**Say:** "Honestly: capacity-limited results, false alarms on empty slices,
2D so no volumetric context, pixel areas can't convert to physical units, only
9 test subjects so variance is high, one split with no cross-validation, and no
clinical validation at all."

## 12. Future work (20 s)

**Say:** "Full-resolution GPU training first — that's the biggest lever.
Then lung-field post-processing for the chest-wall false positives, tuning the
threshold on validation instead of fixing it at 0.5, the attention-U-Net
ablation, and subject-level k-fold."

---

## Closing line

"The thing I'd most want to be judged on isn't the Dice number — it's that the
split is provably leak-free, the empty masks are handled explicitly rather than
hidden, and every number in the repository came from a run I actually
executed."

---

## Questions you should expect, with short answers

**"Your Dice is low."** — "Yes, and I've labelled why: 10 epochs at 128×128 on
one CPU core. It's a lower bound on capacity, not a converged result. The
controlled comparison between the two experiments is still valid because both
ran under identical conditions."

**"Did you run full training?"** — "No. It needs about 16 hours per experiment
on this hardware. I did not report any number for that configuration — the GPU
command is documented in the README."

**"Why is specificity so high everywhere?"** — "Because 99.55% of pixels are
background, so specificity is nearly saturated by construction. That's why I
lead with Dice on foreground-positive slices instead."

**"How do you know augmentation caused the improvement?"** — "I don't isolate
it perfectly — experiment B changes both augmentation and the loss. That's a
fair criticism; the clean ablation would be a third run changing one at a time,
and the framework supports it."

**"What if the masks weren't binary?"** — "The loader validates every sample
and raises on unexpected values. Inspection confirmed all 1,930 masks are
strictly {0, 255}."
