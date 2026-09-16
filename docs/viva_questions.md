# Viva Questions and Answers

Every answer below is consistent with the code in this repository and quotes
only numbers that were actually measured. File references point to where the
behaviour is implemented.

---

## A. Task framing

**1. What is image segmentation and why use it here?**
Segmentation assigns a class label to every pixel, producing an output with the
same spatial resolution as the input. We need it because the clinically useful
quantities — the extent and boundary of a region — only exist at pixel level.

**2. Classification vs detection vs segmentation?**
Classification gives one label per image ("lesion present"). Detection gives a
bounding box per object ("lesion roughly here"). Segmentation gives a
pixel-exact mask ("these exact pixels"). Only segmentation supports area
measurement and boundary analysis. Our model outputs a 1-channel mask the same
size as the input.

**3. Why is this a hard problem in this dataset specifically?**
Three measured reasons: foreground is 0.453% of pixels (220:1 imbalance),
15.80% of masks are completely empty, and 1,930 slices come from only 61
subjects so slices are heavily correlated.

---

## B. Architecture

**4. Why U-Net?**
It was designed for biomedical segmentation with limited annotated data. Its
encoder–decoder shape gives pixel-resolution output, and its skip connections
solve the localisation problem that a plain encoder–decoder has. It trains
end-to-end and is small enough to be reproducible.

**5. Walk me through U-Net.**
Encoder: four stages of `[Conv3×3-BN-ReLU]×2` then 2×2 max-pool, doubling
channels each stage — builds abstract "what" features while discarding "where".
Bottleneck: deepest conv block, lowest resolution, highest channel count.
Decoder: four stages of upsample → concatenate the matching encoder feature map
→ `[Conv3×3-BN-ReLU]×2`. Head: 1×1 conv to 1 channel of raw logits.
(`src/models.py`.)

**6. What does the encoder do?**
Progressively reduces spatial resolution while increasing channel depth,
enlarging the receptive field so deep features describe context and semantics
rather than individual pixels.

**7. What does the decoder do?**
Restores spatial resolution step by step, combining the coarse semantic
features with the fine-detail skip features so the final prediction is both
correctly classified and correctly localised.

**8. What are skip connections and why are they essential?**
They concatenate each encoder feature map onto the decoder stage at the same
resolution. Pooling permanently discards precise spatial detail; upsampling
cannot invent it back. The skips re-inject it. Without them, U-Net produces
blurry, badly localised boundaries — and boundary accuracy is the whole task.

**9. What is convolution doing?**
A small learned kernel slides over the input computing weighted sums, so
features are local and translation-equivariant, and weights are shared across
positions. We use 3×3 kernels with padding 1 so spatial size is preserved
within a block.

**10. What is pooling for?**
2×2 max-pooling halves spatial dimensions, which enlarges the effective
receptive field and cuts computation. The cost is lost spatial precision —
recovered via skip connections.

**11. How do you upsample?**
Configurable (`model.bilinear_upsampling`). Bilinear upsampling by default —
parameter-free and artifact-free; transposed convolution is the learnable
alternative but can produce checkerboard artifacts. Both are tested.

**12. Why is BatchNorm there?**
It normalises activations per channel, stabilising and accelerating training
and reducing sensitivity to initialisation and learning rate.

**13. Why does the model output logits instead of probabilities?**
Numerical stability: `BCEWithLogitsLoss` fuses sigmoid and BCE in a
log-sum-exp-stable way. The sigmoid is applied in the loss and again in the
metric/inference code. A test asserts the output can be negative, catching an
accidentally baked-in sigmoid.

**14. Did you try other architectures?**
Residual U-Net and Attention U-Net are implemented and test-covered
(`MODEL_REGISTRY`). Attention gates are well motivated here since most of every
skip feature map is background. But they were **not** trained on real data, so
they appear in no results table. Plain U-Net is the reported baseline.

---

## C. Loss functions

**15. Explain BCE loss.**
Binary cross-entropy averaged over pixels: `−[y·log p + (1−y)·log(1−p)]`. It
treats every pixel equally, which is exactly its weakness here.

**16. Explain Dice loss.**
`1 − 2|P∩G| / (|P| + |G|)`, a soft overlap measure computed on probabilities.
It depends only on the foreground region, so shrinking the foreground doesn't
shrink its gradient.

**17. Why combine BCE and Dice?**
BCE is stable and well-conditioned but imbalance-blind; Dice is
imbalance-robust but noisy early in training and degenerate on empty masks.
The weighted sum (default 0.5/0.5, configurable) gets both properties.

**18. Prove the imbalance argument.**
`tests/test_losses.py::test_dice_is_more_sensitive_than_bce_under_extreme_imbalance`
constructs a mask with 0.39% foreground; an all-background prediction gives
BCE < 0.1 but Dice > 0.85. And we observed it for real: the BCE-only baseline
collapsed to recall 0.1910 with 41 false-positive pixels in the whole test set.

**19. How do you handle empty masks in the loss?**
Explicitly (`src/losses.py`, `empty_handling`). Default `smooth`: adding `s` to
numerator and denominator makes empty-GT + empty-prediction evaluate to `s/s`
= 1, i.e. zero loss (correct), while empty-GT + non-empty prediction gives
`s/(|P|+s) < 1`, a penalty growing with the false-positive area. The
`exclude` alternative computes Dice only over non-empty-GT samples; it is not
the default because it removes the only signal penalising hallucinated
foreground. Either way, empty samples still contribute to the BCE term — they
are never dropped.

**20. What is class imbalance and how else could you fix it?**
Here, 219.7 background pixels per foreground pixel. Alternatives: `pos_weight`
on BCE (implemented and configurable), focal loss, Tversky/focal-Tversky to
control FP/FN directly, or foreground-biased patch sampling.

---

## D. Metrics

**21. Define Dice and IoU, and how they relate.**
Dice = `2TP / (2TP + FP + FN)`; IoU = `TP / (TP + FP + FN)`. They are monotonic
in each other and Dice ≥ IoU always — asserted by a test over random masks.
Dice weights overlap more generously.

**22. Precision, recall, specificity?**
Precision = `TP/(TP+FP)` — of the pixels I called foreground, how many were.
Recall/sensitivity = `TP/(TP+FN)` — of the true foreground, how much I found.
Specificity = `TN/(TN+FP)` — of the true background, how much I correctly left
alone.

**23. Why is specificity nearly 1.0 for both models?**
Because 99.55% of pixels are background, specificity is saturated by
construction and carries almost no information. That is why we lead with Dice
on foreground-positive slices.

**24. What are HD95 and ASSD, and why pixels not millimetres?**
Surface-distance metrics. ASSD is the mean symmetric distance between the two
boundaries; HD95 is the 95th percentile of the same distance set. We use the
95th percentile rather than the true maximum so a single stray pixel can't
dominate. Pixels, not millimetres, because this dataset has no pixel-spacing
metadata — a millimetre figure would be invented.

**25. What happens to HD95 when a mask is empty?**
It is genuinely undefined — there is no surface. We return `None`, exclude
those cases from the average, and report how many contributed (44 for the
baseline vs 171 for BCE+Dice). We do not substitute 0, which would flatter the
model, or the image diagonal, which would punish it arbitrarily.

**26. Why three metric views instead of one number?**
Because empty-GT slices score a free Dice of 1.0. Our baseline reports 0.1957
"overall" but only **0.0517** on slices that actually contain foreground — the
gap is entirely free points. `positive_only` measures segmentation quality;
`empty_gt_only` measures false alarms; `all` is what the raw test set gives.
Nothing is hidden.

**27. What is foreground coverage?**
The fraction of true foreground pixels recovered — equal to recall at
per-image level. It answers "how much of the structure did we find".

---

## E. Data handling

**28. What is data leakage and why does it matter here?**
Information from evaluation data influencing training. Here the specific risk
is patient-level: ~32 correlated slices per subject, so a random image-level
split puts the same patient in train and test and the model scores well by
recognising memorised anatomy. The metric would then measure memorisation, not
generalisation to a new patient.

**29. How do you prevent it?**
Split by subject, never by image (`src/splitting.py`): 43/9/9 subjects →
1,352/296/282 images. `verify_no_leakage` asserts no shared subject, no shared
file, and that every file's parsed subject belongs to its own split. These run
in the test suite, so a leaked split fails CI. Normalization statistics are
also computed from the training split only — a subtler leakage path.

**30. Why not use the train/val tags already in the filenames?**
They define no test set and we cannot verify they were built subject-wise. We
derive our own split and document the decision.

**31. Validation vs test set?**
Validation drives decisions during training — best-checkpoint selection, LR
scheduling, early stopping. Test is touched once, after all decisions are
frozen, purely to report. If you tune on test, your test number becomes a
training number.

**32. What augmentations did you use and why those?**
Geometric shared by image and mask: horizontal flip, ±10° rotation, ±5%
translation, 0.95–1.05 scale. Photometric on the image only: mild
brightness/contrast and Gaussian noise. They simulate realistic patient
positioning and scanner variation.

**33. Why is vertical flip disabled?**
An axial CT slice has a fixed anterior/posterior orientation. Flipping it
vertically produces an anatomically impossible image the model would never see
at test time, so it would waste capacity. Horizontal flip is kept because
left/right mirroring stays plausible.

**34. How do you guarantee the mask is transformed with the image?**
Both targets go through one `transform(image=..., mask=...)` call so the same
sampled parameters apply, with nearest-neighbour interpolation for the mask.
Crucially, we verify it empirically: `tests/test_augmentations.py` measures
image/mask alignment IoU over 200 random draws and asserts it stays above 0.85,
and separately asserts photometric transforms leave the mask byte-identical.

**35. Why nearest-neighbour resizing for masks?**
Bilinear interpolation blends label values and invents intermediate ones that
are not valid labels. A test shows bilinear resizing of our mask produces 11
distinct values where only 2 are valid; nearest-neighbour produces exactly 2.

**36. Why are masks never normalized?**
A mask is a label field, not a signal. Normalizing it would destroy the {0,1}
semantics and make the loss meaningless.

**37. What does CLAHE do and why is it off by default?**
Contrast Limited Adaptive Histogram Equalization equalises contrast in local
tiles, making low-contrast soft-tissue boundaries more visible, with a clip
limit that stops noise amplification in homogeneous regions. It is off by
default to keep the baseline the simplest reproducible configuration; it is
configurable and applied to images only.

---

## F. Training

**38. Optimizer and learning rate?**
Adam, lr 1e-4, weight decay 1e-5. Adam adapts per-parameter step sizes and is
robust without heavy tuning. 1e-4 is conservative — higher rates destabilise
the Dice term early. AdamW, SGD and RMSprop are also selectable.

**39. What does the LR scheduler do?**
ReduceLROnPlateau halves the LR when validation loss stops improving (patience
3), letting the model take finer steps near a minimum. Cosine and step are also
available. The LR is logged every epoch and plotted.

**40. What is early stopping?**
Training halts when validation Dice hasn't improved for `patience` epochs,
preventing wasted compute and overfitting. Note it monitors **validation**, not
training.

**41. How do you select the best model?**
Highest validation Dice, saved to `<experiment>_best.pt`. A `_last.pt` is
written every epoch for resuming. The test split is never consulted.

**42. What is overfitting and how would you see it here?**
Memorising training data instead of learning generalisable features. The
training-curve figure includes an explicit train-vs-validation gap panel;
a widening gap is the signal. In our run, training Dice reached 0.4557 while
validation was 0.1906 — a gap that would widen further without augmentation.

**43. What does thresholding do?**
Converts the sigmoid probability map to a binary mask at 0.5 by default. It's
the model's operating point: lowering it raises recall and false positives,
raising it does the opposite. It's configurable and exposed as a live slider in
the Streamlit app. We did **not** tune it on validation — a stated limitation.

**44. Mixed precision and resume — did you actually use them?**
Mixed precision activates automatically on CUDA and is a no-op on CPU, so both
paths are identical. Resume was genuinely exercised: the reported `cpu_real`
run was produced across several resumed sessions because of a 300-second
per-command limit, and a test asserts that resuming from epoch 2 with a 4-epoch
target yields a 4-entry history.

---

## G. Results and critique

**45. What are your actual results?**
On the 282-image test split, foreground-positive Dice: baseline (BCE, no
augmentation) **0.0517**; augmented BCE+Dice **0.2351** — a 4.5× improvement.
Best validation Dice 0.1906 and 0.2962 respectively.

**46. Why are the numbers low?**
Compute, stated plainly. Training ran on a single CPU core with no GPU, so we
reduced to 128×128 input, `base_channels=16` and 10 epochs versus the target
256×256 / 32 / 60. That's roughly 16 hours per experiment at target settings.
These are a genuine lower bound, and the A-vs-B comparison is still valid
because both ran under identical conditions.

**47. Did you run full training?**
No. And no number for that configuration appears anywhere in the repository.
The GPU command is documented in the README.

**48. What does your error analysis reveal?**
The baseline collapsed to the majority class: precision 0.9787, specificity
1.0000, but recall 0.1910, with just 41 false-positive pixels in the entire
test set and 192 of 236 positive slices missed entirely. It learned to predict
almost nothing — exactly the failure the Dice term exists to prevent, observed
directly. BCE+Dice cut misses to 65 but false-alarms on 27 of 46 empty slices,
with total FP pixels rising to 12,219 and HD95 to 47.90 px.

**49. Which error is worse?**
A false negative — a missed lesion — is generally more costly in screening than
a false alarm, which a radiologist dismisses. So model B's trade is the more
defensible one. That's why FP and FN are counted separately rather than merged.

**50. What's the most obvious concrete fix?**
The gallery shows a recurring false-positive blob on the left chest wall,
outside the lung field entirely. A lung-field mask or a connected-component
area filter would remove it with no retraining.

**51. How do I know your gallery isn't cherry-picked?**
It's stratified by quartiles of the measured Dice distribution — success,
average, difficult, failure — plus empty-GT cases. The worst cases are included
by construction, whatever the model's quality.

**52. Is your comparison a clean ablation?**
Not perfectly, and that's a fair criticism: experiment B changes both
augmentation and the loss relative to A. A clean ablation needs a third run
varying one factor at a time. The framework supports it; it wasn't run.

**53. CPU vs GPU — why does it matter so much?**
Convolutions are massively parallel; a GPU runs thousands of threads with high
memory bandwidth. Here one epoch took ~72 seconds at 128×128 on one core; at
256×256 with double the width it was ~0.73 s/image/epoch, i.e. hours per
experiment. A GPU would also enable mixed precision and larger batches.

**54. How would you improve performance?**
In order of expected impact: full-resolution GPU training; lung-field
post-processing; threshold tuning on validation; the attention-U-Net ablation;
subject-level k-fold cross-validation; 2.5D inputs for volumetric context; and
Tversky loss for direct FP/FN control.

**55. Biggest limitations?**
Capacity-limited results; false alarms on empty slices; 2D only; pixel areas
can't convert to physical units; 9 test subjects so high variance; unverified
provenance so no benchmark comparison; a single split with no cross-validation;
and no clinical validation of any kind.

**56. How is this reproducible?**
One seed fixes Python/NumPy/PyTorch RNGs and is passed explicitly to the
albumentations `Compose` — without that an augmented run is *not* reproducible
even with torch and numpy seeded, which we found via a test, not by inspection.
The split is a deterministic function of the seed and regenerates the committed
file exactly. Normalization statistics live in the checkpoint. No paths are
hard-coded. A test asserts two identical runs produce identical loss histories.

**57. How do you know the code is correct?**
129 automated tests, all passing with the dataset present. They cover pairing,
mask validation, preprocessing determinism, the nearest-vs-bilinear distinction,
split reproducibility and injected-leakage detection, loss edge cases including
empty masks, metric edge cases and surface-metric undefined cases, forward and
backward passes for all three architectures, checkpoint/resume, and end-to-end
training and evaluation on a synthetic dataset so the suite runs even without
the CT data.
