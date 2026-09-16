"""Loss functions, including explicit empty-mask behaviour."""
import pytest
import torch

from src.losses import DiceLoss, BCEDiceLoss, build_loss, build_loss_by_name
from src.config import load_config

LOGIT_HI, LOGIT_LO = 12.0, -12.0


def logits_like(shape, value):
    return torch.full(shape, value)


def test_dice_loss_near_zero_for_perfect_prediction():
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0
    logits = torch.where(target > 0, LOGIT_HI, LOGIT_LO)
    assert DiceLoss()(logits, target).item() < 0.01


def test_dice_loss_near_one_for_inverted_prediction():
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0
    logits = torch.where(target > 0, LOGIT_LO, LOGIT_HI)
    assert DiceLoss()(logits, target).item() > 0.9


def test_empty_gt_with_empty_prediction_is_rewarded():
    """The central empty-mask case: predicting nothing on an empty mask is correct."""
    target = torch.zeros(2, 1, 16, 16)
    loss = DiceLoss()(logits_like((2, 1, 16, 16), LOGIT_LO), target).item()
    assert loss < 0.05


def test_empty_gt_with_full_prediction_is_penalized():
    """Hallucinating foreground on an empty mask must cost something."""
    target = torch.zeros(2, 1, 16, 16)
    loss = DiceLoss()(logits_like((2, 1, 16, 16), LOGIT_HI), target).item()
    assert loss > 0.9


def test_empty_mask_penalty_scales_with_false_positive_area():
    target = torch.zeros(1, 1, 32, 32)
    small_fp = torch.full((1, 1, 32, 32), LOGIT_LO)
    small_fp[:, :, :4, :4] = LOGIT_HI
    big_fp = torch.full((1, 1, 32, 32), LOGIT_LO)
    big_fp[:, :, :20, :20] = LOGIT_HI
    d = DiceLoss()
    assert d(big_fp, target).item() > d(small_fp, target).item()


def test_exclude_mode_returns_zero_for_all_empty_batch():
    target = torch.zeros(2, 1, 8, 8)
    loss = DiceLoss(empty_handling="exclude")(logits_like((2, 1, 8, 8), LOGIT_HI), target)
    assert loss.item() == 0.0


def test_exclude_mode_uses_only_non_empty_samples():
    target = torch.zeros(2, 1, 8, 8)
    target[1, :, 2:6, 2:6] = 1.0
    logits = torch.where(target > 0, LOGIT_HI, LOGIT_LO)
    # Sample 0 is empty-GT/empty-pred; sample 1 is perfect. Both give ~0 loss.
    assert DiceLoss(empty_handling="exclude")(logits, target).item() < 0.05


def test_invalid_empty_handling_raises():
    with pytest.raises(ValueError):
        DiceLoss(empty_handling="nonsense")


def test_bce_dice_respects_weights():
    target = torch.rand(2, 1, 16, 16).round()
    logits = torch.randn(2, 1, 16, 16)
    only_bce = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)(logits, target)
    only_dice = BCEDiceLoss(bce_weight=0.0, dice_weight=1.0)(logits, target)
    both = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)(logits, target)
    assert torch.allclose(both, 0.5 * only_bce + 0.5 * only_dice, atol=1e-5)


def test_all_losses_are_finite_and_differentiable():
    cfg = load_config()
    target = torch.rand(2, 1, 16, 16).round()
    for name in ("bce", "dice", "bce_dice"):
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        loss = build_loss_by_name(name, cfg)(logits, target)
        assert torch.isfinite(loss), name
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all(), name


def test_build_loss_uses_config_name():
    cfg = load_config()
    assert isinstance(build_loss(cfg), BCEDiceLoss)


def test_unknown_loss_name_raises():
    with pytest.raises(ValueError):
        build_loss_by_name("triple_focal_supreme", load_config())


def test_dice_is_more_sensitive_than_bce_under_extreme_imbalance():
    """
    Motivating check for the BCE+Dice choice: with ~0.4% foreground, a model
    that predicts all-background gets a small BCE but a maximal Dice loss.
    """
    target = torch.zeros(1, 1, 64, 64)
    target[:, :, :4, :4] = 1.0  # 16/4096 = 0.39% foreground
    all_background = torch.full((1, 1, 64, 64), LOGIT_LO)
    bce = torch.nn.BCEWithLogitsLoss()(all_background, target).item()
    dice = DiceLoss()(all_background, target).item()
    assert bce < 0.1, "BCE should barely notice the missed foreground"
    assert dice > 0.85, "Dice should heavily penalize the missed foreground"
