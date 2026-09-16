"""
Loss functions for binary segmentation of imbalanced medical masks.

Why Dice-based loss for this dataset
------------------------------------
Dataset inspection (outputs/metrics/dataset_inspection_report.md) measured a
mean foreground fraction of ~0.45% of pixels per mask, with ~15.8% of masks
entirely empty. That means a trivial model predicting "background everywhere"
already achieves ~99.5% pixel accuracy and a very low BCE. Plain BCE averages
over pixels, so the ~99.55% background pixels dominate the gradient and the
sparse foreground contributes almost nothing; training converges to the
degenerate all-background solution.

Dice loss instead optimizes 1 - (2|P∩G|)/(|P|+|G|), a *region overlap* term
whose value depends only on the foreground. Scaling the foreground down does
not shrink the Dice gradient the way it shrinks the BCE gradient, so Dice is
far more robust to imbalance. Its weaknesses are the mirror image: it gives
noisy gradients early in training (when P is near-random the ratio is
unstable) and it is undefined as 0/0 when both prediction and ground truth
are empty.

Combining them — L = w_bce * BCE + w_dice * Dice — keeps BCE's stable,
well-conditioned per-pixel signal while letting Dice supply the
imbalance-robust overlap signal. Weights are configurable (loss.bce_weight /
loss.dice_weight in config.yaml).

Empty-mask handling (explicit, not incidental)
----------------------------------------------
~15.8% of this dataset's masks contain no foreground at all. For such a
sample, intersection = 0 and |P| + |G| = |P|, so the raw Dice ratio is 0/0
when the model also predicts nothing. Two strategies are supported via
`empty_handling`:

  "smooth" (default) - add `smooth` to numerator and denominator:
        (2*TP + s) / (|P| + |G| + s). With an empty GT and an empty
        prediction this evaluates to s/s = 1, i.e. loss 0 — correct, the model
        got it right. With an empty GT and a non-empty prediction it evaluates
        to s/(|P| + s) < 1, i.e. a positive loss that grows with the size of
        the false positive. This is the behaviour we want and is what the
        project uses.

  "exclude" - compute the Dice term only over samples whose ground truth is
        non-empty, and fall back to the BCE term alone for empty-GT samples.
        Provided for ablation; it is NOT the default, because it removes the
        only signal that penalizes hallucinated foreground on empty slices.

Whichever is chosen, empty-GT samples are never silently dropped from the
batch: they always contribute to the BCE term.
"""
import torch
import torch.nn as nn

VALID_EMPTY_HANDLING = ("smooth", "exclude")


class DiceLoss(nn.Module):
    """Soft (probability-based) Dice loss, computed per-sample then averaged."""

    def __init__(self, smooth: float = 1.0, empty_handling: str = "smooth"):
        super().__init__()
        if empty_handling not in VALID_EMPTY_HANDLING:
            raise ValueError(
                f"empty_handling must be one of {VALID_EMPTY_HANDLING}, got '{empty_handling}'"
            )
        self.smooth = smooth
        self.empty_handling = empty_handling

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).reshape(logits.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        cardinality = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        per_sample_loss = 1 - dice

        if self.empty_handling == "exclude":
            non_empty = (targets.sum(dim=1) > 0)
            if non_empty.any():
                return per_sample_loss[non_empty].mean()
            # Entire batch has empty ground truth: no Dice term is defined.
            return logits.sum() * 0.0

        return per_sample_loss.mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and soft Dice."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 dice_smooth: float = 1.0, empty_handling: str = "smooth",
                 pos_weight: float = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        pw = None if pos_weight in (None, 0) else torch.tensor([float(pos_weight)])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = DiceLoss(smooth=dice_smooth, empty_handling=empty_handling)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def _bce(cfg):
    pw = cfg.loss.get("pos_weight", None)
    pos_weight = None if pw in (None, 0) else torch.tensor([float(pw)])
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def build_loss_by_name(name: str, cfg):
    """Build a loss by explicit name ('bce', 'dice', 'bce_dice')."""
    lc = cfg.loss
    empty_handling = lc.get("empty_handling", "smooth")
    if name == "bce":
        return _bce(cfg)
    if name == "dice":
        return DiceLoss(smooth=lc.dice_smooth, empty_handling=empty_handling)
    if name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=lc.bce_weight,
            dice_weight=lc.dice_weight,
            dice_smooth=lc.dice_smooth,
            empty_handling=empty_handling,
            pos_weight=lc.get("pos_weight", None),
        )
    raise ValueError(f"Unknown loss name: {name}. Valid: 'bce', 'dice', 'bce_dice'")


def build_loss(cfg):
    """Build the loss named in config.yaml (loss.name)."""
    return build_loss_by_name(cfg.loss.name, cfg)
