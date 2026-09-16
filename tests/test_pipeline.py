"""
End-to-end pipeline tests.

`test_full_training_run_on_synthetic_data` runs the real `train_model` entry
point on the synthetic dataset — dataloaders, augmentation, model, loss,
backward pass, scheduler, checkpointing, early stopping and resume — and then
evaluates the resulting checkpoint. It is the test that proves the pipeline
actually works end to end on a machine with no GPU and no CT dataset.
"""
import copy
import json
import os

import numpy as np
import pytest
import torch

from src.config import load_config, resolve_path
from src.train import build_dataloaders, train_model, run_epoch, build_optimizer, build_scheduler
from src.models import build_model
from src.losses import build_loss
from src.evaluate import evaluate_checkpoint, resolve_checkpoint_path
from src.error_analysis import analyze
from src.splitting import build_split
from src.utils import get_device, set_seed


@pytest.fixture
def synth_cfg(synthetic_dataset, tmp_path):
    """A config pointed at the synthetic dataset, with tiny training settings."""
    cfg = load_config()
    cfg.data["images_dir"] = synthetic_dataset["images_dir"]
    cfg.data["masks_dir"] = synthetic_dataset["masks_dir"]
    cfg.data["image_size"] = 64
    cfg.model["base_channels"] = 8
    cfg.training["device"] = "cpu"
    cfg.training["early_stopping_patience"] = 10
    cfg.training.smoke_test["epochs"] = 2
    cfg.training.smoke_test["batch_size"] = 4
    cfg.training.smoke_test["max_train_samples"] = 12
    cfg.training.smoke_test["max_val_samples"] = 6
    cfg.paths["checkpoints_dir"] = str(tmp_path / "ckpt")
    return cfg


@pytest.fixture
def synth_split(synthetic_dataset):
    return build_split(synthetic_dataset["images_dir"], 0.2, 0.2, seed=42)


def test_absolute_paths_in_config_are_honoured(synth_cfg, synth_split):
    train_loader, val_loader, stats = build_dataloaders(synth_cfg, synth_split)
    assert len(train_loader.dataset) > 0 and len(val_loader.dataset) > 0
    assert 0.0 <= stats["mean"] <= 1.0


def test_single_training_step_produces_finite_loss(synth_cfg, synth_split):
    set_seed(synth_cfg.seed)
    train_loader, _, _ = build_dataloaders(synth_cfg, synth_split)
    device = get_device("cpu")
    model = build_model(synth_cfg).to(device)
    criterion = build_loss(synth_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch = next(iter(train_loader))
    loss = criterion(model(batch["image"].to(device)), batch["mask"].to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss).item()


def test_run_epoch_reports_loss_and_metrics(synth_cfg, synth_split):
    _, val_loader, _ = build_dataloaders(synth_cfg, synth_split)
    model = build_model(synth_cfg)
    agg = run_epoch(model, val_loader, build_loss(synth_cfg), get_device("cpu"))
    assert np.isfinite(agg["loss"])
    assert 0.0 <= agg["dice"] <= 1.0
    assert agg["n_samples"] == len(val_loader.dataset)


@pytest.mark.parametrize("opt", ["adam", "adamw", "sgd", "rmsprop"])
def test_every_optimizer_builds(synth_cfg, opt):
    cfg = copy.deepcopy(synth_cfg)
    cfg.training["optimizer"] = opt
    assert build_optimizer(cfg, build_model(cfg)) is not None


@pytest.mark.parametrize("sched", ["none", "reduce_on_plateau", "cosine", "step"])
def test_every_scheduler_builds(synth_cfg, sched):
    cfg = copy.deepcopy(synth_cfg)
    cfg.training["lr_scheduler"] = sched
    scheduler, on_metric = build_scheduler(cfg, build_optimizer(cfg, build_model(cfg)), 5)
    assert (scheduler is None) == (sched == "none")
    assert on_metric == (sched == "reduce_on_plateau")


def test_unknown_optimizer_and_scheduler_raise(synth_cfg):
    cfg = copy.deepcopy(synth_cfg)
    cfg.training["optimizer"] = "nope"
    with pytest.raises(ValueError):
        build_optimizer(cfg, build_model(cfg))
    cfg.training["optimizer"] = "adam"
    cfg.training["lr_scheduler"] = "nope"
    with pytest.raises(ValueError):
        build_scheduler(cfg, build_optimizer(cfg, build_model(cfg)), 5)


def test_full_training_run_on_synthetic_data(synth_cfg, synth_split):
    result = train_model(synth_cfg, synth_split, mode="smoke_test",
                         experiment_name="pytest_synth", log_fn=lambda *a: None)

    assert result["epochs_run"] == 2
    for key in ("train_loss", "val_loss", "train_dice", "val_dice",
                "train_iou", "val_iou", "learning_rate"):
        assert len(result["history"][key]) == 2, key
        assert all(np.isfinite(v) for v in result["history"][key]), key

    ckpt_dir = synth_cfg.paths.checkpoints_dir
    assert os.path.exists(os.path.join(ckpt_dir, "pytest_synth_best.pt"))
    assert os.path.exists(os.path.join(ckpt_dir, "pytest_synth_last.pt"))


def test_checkpoint_contains_everything_needed_to_resume(synth_cfg, synth_split):
    train_model(synth_cfg, synth_split, mode="smoke_test",
                experiment_name="pytest_ckpt", log_fn=lambda *a: None)
    ckpt = torch.load(os.path.join(synth_cfg.paths.checkpoints_dir, "pytest_ckpt_last.pt"),
                      map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "epoch",
                "normalize_stats", "history", "model_name"):
        assert key in ckpt, key
    assert 0.0 <= ckpt["normalize_stats"]["mean"] <= 1.0


def test_resume_continues_from_the_saved_epoch(synth_cfg, synth_split):
    train_model(synth_cfg, synth_split, mode="smoke_test",
                experiment_name="pytest_resume", log_fn=lambda *a: None)
    last = os.path.join(synth_cfg.paths.checkpoints_dir, "pytest_resume_last.pt")

    longer = copy.deepcopy(synth_cfg)
    longer.training.smoke_test["epochs"] = 4
    resumed = train_model(longer, synth_split, mode="smoke_test",
                          experiment_name="pytest_resume", resume_from=last,
                          log_fn=lambda *a: None)
    # 2 epochs before + 2 after = 4 entries of history, not 2.
    assert len(resumed["history"]["train_loss"]) == 4


def test_resume_from_missing_checkpoint_raises(synth_cfg, synth_split):
    with pytest.raises(FileNotFoundError):
        train_model(synth_cfg, synth_split, mode="smoke_test",
                    experiment_name="x", resume_from="/nonexistent/ckpt.pt",
                    log_fn=lambda *a: None)


def test_evaluation_of_a_trained_checkpoint(synth_cfg, synth_split):
    train_model(synth_cfg, synth_split, mode="smoke_test",
                experiment_name="pytest_eval", log_fn=lambda *a: None)
    ckpt = os.path.join(synth_cfg.paths.checkpoints_dir, "pytest_eval_best.pt")

    result = evaluate_checkpoint(synth_cfg, ckpt, synth_split["test_files"],
                                 split_name="test", batch_size=4)
    sg = result["subgroup_metrics"]
    assert sg["n_all"] == len(synth_split["test_files"])
    assert sg["n_positive_gt"] + sg["n_empty_gt"] == sg["n_all"]
    assert len(result["per_sample_metrics"]) == sg["n_all"]
    for m in result["per_sample_metrics"]:
        assert 0.0 <= m["dice"] <= 1.0
        assert m["filename"] and m["subject_id"]


def test_prediction_dimensions_match_the_input(synth_cfg, synth_split, synthetic_dataset):
    from src.predict import load_model_for_inference, predict_from_config
    from src.preprocessing import load_image_grayscale

    train_model(synth_cfg, synth_split, mode="smoke_test",
                experiment_name="pytest_pred", log_fn=lambda *a: None)
    ckpt = os.path.join(synth_cfg.paths.checkpoints_dir, "pytest_pred_best.pt")
    model, stats, device = load_model_for_inference(ckpt, synth_cfg)

    img = load_image_grayscale(os.path.join(synthetic_dataset["images_dir"],
                                            synthetic_dataset["filenames"][0]))
    prob, pred, resized = predict_from_config(model, stats, device, img, synth_cfg)
    size = synth_cfg.data.image_size
    assert prob.shape == (size, size) and pred.shape == (size, size)
    assert resized.shape == (size, size)
    assert set(np.unique(pred).tolist()).issubset({0, 1})
    assert prob.min() >= 0.0 and prob.max() <= 1.0


def test_prediction_works_on_an_odd_sized_input(synth_cfg, synth_split):
    from src.predict import load_model_for_inference, predict_from_config
    train_model(synth_cfg, synth_split, mode="smoke_test",
                experiment_name="pytest_odd", log_fn=lambda *a: None)
    model, stats, device = load_model_for_inference(
        os.path.join(synth_cfg.paths.checkpoints_dir, "pytest_odd_best.pt"), synth_cfg)
    odd = np.random.default_rng(0).integers(0, 256, size=(97, 133), dtype=np.uint8)
    prob, pred, _ = predict_from_config(model, stats, device, odd, synth_cfg)
    assert prob.shape == (synth_cfg.data.image_size, synth_cfg.data.image_size)


def test_error_analysis_runs_on_real_evaluation_output(synth_cfg, synth_split):
    train_model(synth_cfg, synth_split, mode="smoke_test",
                experiment_name="pytest_ea", log_fn=lambda *a: None)
    ckpt = os.path.join(synth_cfg.paths.checkpoints_dir, "pytest_ea_best.pt")
    result = evaluate_checkpoint(synth_cfg, ckpt, synth_split["test_files"], batch_size=4)
    analysis = analyze(result["per_sample_metrics"], synth_cfg)
    assert analysis["n_samples"] == len(synth_split["test_files"])
    assert analysis["n_gt_positive"] + analysis["n_gt_empty"] == analysis["n_samples"]
    assert isinstance(analysis["category_counts"], dict)


def test_checkpoint_path_resolution_is_portable():
    """Result files may carry a path from another machine — resolution must cope."""
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_path("/some/machine/that/does/not/exist/model_best.pt")


def test_training_is_reproducible_from_the_seed(synth_cfg, synth_split):
    a = train_model(synth_cfg, synth_split, mode="smoke_test",
                    experiment_name="pytest_seed_a", log_fn=lambda *a: None)
    b = train_model(synth_cfg, synth_split, mode="smoke_test",
                    experiment_name="pytest_seed_b", log_fn=lambda *a: None)
    assert a["history"]["train_loss"] == pytest.approx(b["history"]["train_loss"], abs=1e-6)
