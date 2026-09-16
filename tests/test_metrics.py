"""Segmentation metrics, edge cases, surface distances and subgroup reporting."""
import numpy as np
import pytest

from src.metrics import (
    compute_all_metrics, aggregate_metrics, aggregate_by_subgroup,
    confusion_components, hd95, assd, logits_to_binary,
)


def box(size=32, y0=8, y1=20, x0=8, x1=20):
    m = np.zeros((size, size), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def test_perfect_prediction():
    gt = box()
    m = compute_all_metrics(gt.copy(), gt)
    for k in ("dice", "iou", "precision", "recall", "specificity", "f1"):
        assert abs(m[k] - 1.0) < 1e-6, k


def test_empty_gt_and_empty_pred_scores_perfect():
    z = np.zeros((10, 10), dtype=np.uint8)
    m = compute_all_metrics(z, z)
    assert m["dice"] == 1.0 and m["iou"] == 1.0
    assert m["tp"] == 0 and m["fp"] == 0 and m["fn"] == 0


def test_disjoint_prediction_scores_zero():
    gt = np.zeros((10, 10), dtype=np.uint8); gt[0:5, :] = 1
    pred = np.zeros((10, 10), dtype=np.uint8); pred[5:, :] = 1
    m = compute_all_metrics(pred, gt)
    assert m["dice"] == 0.0 and m["iou"] == 0.0 and m["tp"] == 0


def test_confusion_components_sum_to_total():
    gt, pred = box(), box(y0=10, y1=24, x0=10, x1=24)
    tp, tn, fp, fn = confusion_components(pred, gt)
    assert tp + tn + fp + fn == gt.size


def test_dice_matches_hand_computation():
    gt, pred = box(), box(y0=10, y1=22, x0=10, x1=22)
    tp, _, fp, fn = confusion_components(pred, gt)
    expected = 2 * tp / (2 * tp + fp + fn)
    assert abs(compute_all_metrics(pred, gt)["dice"] - expected) < 1e-6


def test_dice_is_always_at_least_iou():
    rng = np.random.default_rng(3)
    for _ in range(20):
        gt = (rng.random((24, 24)) > 0.7).astype(np.uint8)
        pred = (rng.random((24, 24)) > 0.7).astype(np.uint8)
        m = compute_all_metrics(pred, gt)
        assert m["dice"] >= m["iou"] - 1e-9


def test_metrics_stay_in_unit_range():
    rng = np.random.default_rng(4)
    for _ in range(20):
        gt = (rng.random((16, 16)) > 0.5).astype(np.uint8)
        pred = (rng.random((16, 16)) > 0.5).astype(np.uint8)
        for k, v in compute_all_metrics(pred, gt).items():
            if k in ("dice", "iou", "precision", "recall", "specificity", "f1",
                     "foreground_coverage"):
                assert 0.0 <= v <= 1.0, (k, v)


def test_missed_positive_gives_zero_recall():
    m = compute_all_metrics(np.zeros((16, 16), dtype=np.uint8), box(16, 2, 8, 2, 8))
    assert m["recall"] == 0.0 and m["dice"] == 0.0


def test_false_alarm_on_empty_gt_gives_zero_precision():
    m = compute_all_metrics(box(16, 2, 8, 2, 8), np.zeros((16, 16), dtype=np.uint8))
    assert m["precision"] == 0.0 and m["specificity"] < 1.0


def test_foreground_fractions_are_reported():
    gt = box()
    m = compute_all_metrics(gt.copy(), gt)
    assert abs(m["gt_foreground_fraction"] - gt.mean()) < 1e-6
    assert abs(m["pred_foreground_fraction"] - gt.mean()) < 1e-6


# --- surface metrics -------------------------------------------------------

def test_surface_distance_zero_for_identical_masks():
    gt = box()
    assert hd95(gt.copy(), gt) == 0.0
    assert assd(gt.copy(), gt) == 0.0


def test_surface_distance_grows_with_displacement():
    gt = box()
    near = box(y0=9, y1=21, x0=9, x1=21)
    far = box(y0=16, y1=28, x0=16, x1=28)
    assert hd95(far, gt) > hd95(near, gt)
    assert assd(far, gt) > assd(near, gt)


def test_surface_distance_undefined_when_one_mask_empty():
    assert hd95(np.zeros((16, 16), dtype=np.uint8), box(16, 2, 8, 2, 8)) is None
    assert assd(box(16, 2, 8, 2, 8), np.zeros((16, 16), dtype=np.uint8)) is None


def test_surface_metrics_included_only_on_request():
    gt = box()
    assert "hd95" not in compute_all_metrics(gt.copy(), gt)
    assert "hd95" in compute_all_metrics(gt.copy(), gt, include_surface=True)


# --- aggregation -----------------------------------------------------------

def test_aggregate_skips_undefined_surface_metrics():
    gt = box()
    defined = compute_all_metrics(gt.copy(), gt, include_surface=True)
    undefined = compute_all_metrics(np.zeros((32, 32), dtype=np.uint8),
                                    np.zeros((32, 32), dtype=np.uint8),
                                    include_surface=True)
    agg = aggregate_metrics([defined, undefined])
    assert agg["n_hd95_defined"] == 1
    assert agg["hd95"] == 0.0


def test_aggregate_of_empty_list_does_not_crash():
    agg = aggregate_metrics([])
    assert agg["n_samples"] == 0 and agg["dice"] == 0.0


def test_subgroups_partition_the_samples():
    gt = box()
    positive = compute_all_metrics(gt.copy(), gt)
    empty = compute_all_metrics(np.zeros((32, 32), dtype=np.uint8),
                                np.zeros((32, 32), dtype=np.uint8))
    sg = aggregate_by_subgroup([positive, empty, positive])
    assert sg["n_all"] == 3
    assert sg["n_positive_gt"] + sg["n_empty_gt"] == sg["n_all"]
    assert sg["n_positive_gt"] == 2 and sg["n_empty_gt"] == 1


def test_empty_masks_inflate_the_all_subgroup():
    """The documented reason the report separates 'all' from 'positive only'."""
    gt = box()
    mediocre = compute_all_metrics(box(32, 14, 26, 14, 26), gt)
    free_point = compute_all_metrics(np.zeros((32, 32), dtype=np.uint8),
                                     np.zeros((32, 32), dtype=np.uint8))
    sg = aggregate_by_subgroup([mediocre, free_point])
    assert sg["all"]["dice"] > sg["positive_only"]["dice"]


def test_logits_to_binary_respects_threshold():
    import torch
    logits = torch.tensor([[[[-2.0, 2.0]]]])
    assert logits_to_binary(logits, 0.5).tolist() == [[[[0, 1]]]]
    assert logits_to_binary(logits, 0.05).tolist() == [[[[1, 1]]]]
