"""Training and validation loops for the U-Net segmentation model."""
import os
import time
import torch
from torch.utils.data import DataLoader

from .dataset import LungSegmentationDataset, compute_dataset_stats
from .models import build_model
from .losses import build_loss_by_name
from .augmentations import get_train_augmentations, get_val_augmentations
from .metrics import compute_all_metrics, aggregate_metrics, logits_to_binary
from .utils import get_device, ensure_dir, count_parameters
from .config import resolve_path, get_project_root


def build_optimizer(cfg, model):
    """Build the optimizer named in config (training.optimizer)."""
    name = str(cfg.training.optimizer).lower()
    params = model.parameters()
    lr, wd = cfg.training.learning_rate, cfg.training.weight_decay
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd,
                               momentum=cfg.training.get("momentum", 0.9))
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer '{name}'. Valid: adam, adamw, sgd, rmsprop")


def build_scheduler(cfg, optimizer, epochs):
    """
    Build the LR scheduler named in config (training.lr_scheduler).

    Returns (scheduler, steps_on_metric). `steps_on_metric` is True for
    ReduceLROnPlateau, which needs the validation loss passed to .step().
    """
    name = str(cfg.training.lr_scheduler).lower()
    if name in ("none", "off", ""):
        return None, False
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=cfg.training.get("scheduler_patience", 3)), True
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)), False
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.training.get("scheduler_step_size", 10),
            gamma=0.5), False
    raise ValueError(f"Unknown lr_scheduler '{name}'. "
                     f"Valid: none, reduce_on_plateau, cosine, step")


def build_dataloaders(cfg, split, mode="smoke_test", use_augmentation=True):
    """
    Build train/val DataLoaders for the given split dict.

    Normalization statistics are computed from the TRAINING files only, so no
    validation or test intensity information reaches the preprocessing step.
    """
    images_dir = resolve_path(cfg.data.images_dir)
    masks_dir = resolve_path(cfg.data.masks_dir)

    train_files = split["train_files"]
    val_files = split["val_files"]

    tr_cfg = getattr(cfg.training, mode)
    max_train = tr_cfg.get("max_train_samples") if mode == "smoke_test" else None
    max_val = tr_cfg.get("max_val_samples") if mode == "smoke_test" else None

    # Statistics come from the files actually used for training.
    stat_files = train_files[:max_train] if max_train else train_files
    mean, std = compute_dataset_stats(
        stat_files, images_dir, image_size=cfg.data.image_size,
        max_samples=min(500, len(stat_files)),
        use_clahe=cfg.data.get("use_clahe", False),
        clahe_clip_limit=cfg.data.get("clahe_clip_limit", 2.0),
        clahe_tile_grid=cfg.data.get("clahe_tile_grid", 8),
    )

    train_transform = (get_train_augmentations(cfg)
                       if (cfg.augmentation.enabled and use_augmentation) else None)
    val_transform = get_val_augmentations()

    train_ds = LungSegmentationDataset.from_config(
        cfg, train_files, images_dir, masks_dir, transform=train_transform,
        normalize_stats=(mean, std), max_samples=max_train)
    val_ds = LungSegmentationDataset.from_config(
        cfg, val_files, images_dir, masks_dir, transform=val_transform,
        normalize_stats=(mean, std), max_samples=max_val)

    nw = tr_cfg.get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=tr_cfg.batch_size, shuffle=True,
                              num_workers=nw, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=tr_cfg.batch_size, shuffle=False,
                            num_workers=nw)
    return train_loader, val_loader, {"mean": mean, "std": std}


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              grad_clip=None):
    """
    Run one pass over `loader`. Training if `optimizer` is given, else eval.

    Mixed precision is used when `scaler` is enabled (CUDA only); on CPU the
    autocast context is a no-op and the scaler is disabled, so the same code
    path runs correctly on both.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    use_amp = scaler is not None and scaler.is_enabled()
    total_loss, n_batches = 0.0, 0
    all_metrics = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            preds_bin = logits_to_binary(logits.float())
            gts_bin = masks.detach().cpu().numpy().astype("uint8")
            for i in range(preds_bin.shape[0]):
                all_metrics.append(compute_all_metrics(preds_bin[i, 0], gts_bin[i, 0]))

    agg = aggregate_metrics(all_metrics)
    agg["loss"] = total_loss / max(1, n_batches)
    return agg


def _save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_dice,
                     cfg, norm_stats, history, experiment_name, model_name):
    """Save a full, resumable checkpoint."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "val_dice": best_val_dice,
        "config_model": dict(cfg.model),
        "model_name": model_name,
        "normalize_stats": norm_stats,
        "history": history,
        "experiment_name": experiment_name,
    }, path)


def train_model(cfg, split, mode="smoke_test", experiment_name="default",
                augmentation_override=None, loss_override=None,
                model_override=None, resume_from=None, log_fn=print):
    """
    Train the model and return the training history plus checkpoint paths.

    `mode` is "smoke_test" or "full", controlling epoch count / batch size /
    sample caps as defined in config.yaml. Model selection is by best
    VALIDATION Dice — the test split is never consulted during training.
    """
    from .utils import set_seed
    set_seed(cfg.seed)

    use_aug = cfg.augmentation.enabled if augmentation_override is None else augmentation_override
    train_loader, val_loader, norm_stats = build_dataloaders(
        cfg, split, mode=mode, use_augmentation=use_aug)

    device = get_device(cfg.training.device)
    model_name = model_override or cfg.model.get("name", "unet")
    model = build_model(cfg, name_override=model_name).to(device)

    loss_name = loss_override or cfg.loss.name
    criterion = build_loss_by_name(loss_name, cfg).to(device)

    tr_cfg = getattr(cfg.training, mode)
    epochs = tr_cfg.epochs

    optimizer = build_optimizer(cfg, model)
    scheduler, sched_on_metric = build_scheduler(cfg, optimizer, epochs)

    # Mixed precision only makes sense on CUDA; enabled=False makes every AMP
    # call a no-op so the CPU path is byte-identical to non-AMP training.
    amp_requested = bool(cfg.training.get("mixed_precision", True))
    use_amp = amp_requested and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    grad_clip = cfg.training.get("grad_clip_norm", None)

    ckpt_dir = (cfg.paths.checkpoints_dir if os.path.isabs(cfg.paths.checkpoints_dir)
                else os.path.join(get_project_root(), cfg.paths.checkpoints_dir))
    ensure_dir(ckpt_dir)
    best_ckpt_path = os.path.join(ckpt_dir, f"{experiment_name}_best.pt")
    last_ckpt_path = os.path.join(ckpt_dir, f"{experiment_name}_last.pt")

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [],
               "train_iou": [], "val_iou": [], "learning_rate": []}
    best_val_dice = -1.0
    start_epoch = 1
    no_improve_epochs = 0

    # ---- optional resume -------------------------------------------------
    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        history = ckpt.get("history", history)
        for k in ("train_loss", "val_loss", "train_dice", "val_dice",
                  "train_iou", "val_iou", "learning_rate"):
            history.setdefault(k, [])
        best_val_dice = ckpt.get("val_dice", -1.0)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        log_fn(f"[{experiment_name}] Resumed from {resume_from} at epoch {start_epoch} "
               f"(best_val_dice={best_val_dice:.4f})")

    n_params = count_parameters(model)
    log_fn(f"[{experiment_name}] model={model_name} device={device} params={n_params:,} "
           f"epochs={epochs} batch_size={tr_cfg.batch_size} amp={use_amp} "
           f"train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} "
           f"augmentation={use_aug} loss={loss_name}")

    start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = run_epoch(model, train_loader, criterion, device,
                                  optimizer=optimizer, scaler=scaler, grad_clip=grad_clip)
        val_metrics = run_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_metrics["loss"]) if sched_on_metric else scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(val_metrics["dice"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["learning_rate"].append(current_lr)

        log_fn(f"[{experiment_name}] epoch {epoch}/{epochs} "
               f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
               f"train_dice={train_metrics['dice']:.4f} val_dice={val_metrics['dice']:.4f} "
               f"lr={current_lr:.2e} ({time.time() - epoch_start:.1f}s)")

        improved = val_metrics["dice"] > best_val_dice
        if improved:
            best_val_dice = val_metrics["dice"]
            no_improve_epochs = 0
            _save_checkpoint(best_ckpt_path, model, optimizer, scheduler, scaler, epoch,
                             best_val_dice, cfg, norm_stats, history, experiment_name, model_name)
        else:
            no_improve_epochs += 1

        # "latest" checkpoint is written every epoch so training can resume
        # after an interruption, independent of validation performance.
        _save_checkpoint(last_ckpt_path, model, optimizer, scheduler, scaler, epoch,
                         best_val_dice, cfg, norm_stats, history, experiment_name, model_name)

        if no_improve_epochs >= cfg.training.early_stopping_patience:
            log_fn(f"[{experiment_name}] Early stopping at epoch {epoch} "
                   f"(no val improvement for {no_improve_epochs} epochs).")
            break

    return {
        "experiment_name": experiment_name,
        "mode": mode,
        "model": model_name,
        "augmentation": use_aug,
        "loss": loss_name,
        "optimizer": cfg.training.optimizer,
        "lr_scheduler": cfg.training.lr_scheduler,
        "mixed_precision": use_amp,
        "epochs_run": len(history["train_loss"]),
        "best_val_dice": best_val_dice,
        # Stored RELATIVE to the project root so results files stay valid when
        # the project is copied to another machine.
        "best_checkpoint": os.path.relpath(best_ckpt_path, get_project_root()),
        "last_checkpoint": os.path.relpath(last_ckpt_path, get_project_root()),
        "total_train_time_sec": time.time() - start_time,
        "n_params": n_params,
        "device": str(device),
        "history": history,
        "normalize_stats": norm_stats,
    }
