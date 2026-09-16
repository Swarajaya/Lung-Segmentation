"""Plotting utilities: training curves, per-sample panels, overlays."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(history, out_path, title_prefix=""):
    """
    Publication-style 2x3 grid: loss, Dice, IoU, learning rate, plus a
    train-vs-val generalization-gap panel that makes overfitting visible at a
    glance (a widening gap = the model is memorising the training subjects).
    """
    has_lr = bool(history.get("learning_rate"))
    fig, axes_grid = plt.subplots(2, 3, figsize=(16, 8.5))
    axes = axes_grid.ravel()
    epochs_x = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs_x, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs_x, history["val_loss"], marker="s", label="validation")
    axes[0].set_title(f"{title_prefix}Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_x, history["train_dice"], marker="o", label="train")
    axes[1].plot(epochs_x, history["val_dice"], marker="s", label="validation")
    axes[1].set_title(f"{title_prefix}Dice coefficient")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs_x, history["train_iou"], marker="o", label="train")
    axes[2].plot(epochs_x, history["val_iou"], marker="s", label="validation")
    axes[2].set_title(f"{title_prefix}IoU")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    # --- learning rate ---
    if has_lr:
        axes[3].plot(epochs_x, history["learning_rate"], marker="d", color="tab:green")
        axes[3].set_yscale("log")
        axes[3].set_title(f"{title_prefix}Learning rate")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("LR (log scale)")
    else:
        axes[3].text(0.5, 0.5, "Learning rate not recorded\n(checkpoint predates LR logging)",
                     ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_title(f"{title_prefix}Learning rate")
    axes[3].grid(alpha=0.3)

    # --- validation Dice vs IoU together ---
    axes[4].plot(epochs_x, history["val_dice"], marker="s", label="val Dice")
    axes[4].plot(epochs_x, history["val_iou"], marker="^", label="val IoU")
    axes[4].set_title(f"{title_prefix}Validation Dice & IoU")
    axes[4].set_xlabel("Epoch")
    axes[4].set_ylabel("Score")
    axes[4].legend()
    axes[4].grid(alpha=0.3)

    # --- generalization gap ---
    gap_loss = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
    gap_dice = [t - v for t, v in zip(history["train_dice"], history["val_dice"])]
    axes[5].plot(epochs_x, gap_loss, marker="o", label="val loss - train loss")
    axes[5].plot(epochs_x, gap_dice, marker="s", label="train Dice - val Dice")
    axes[5].axhline(0, color="gray", lw=1, ls="--")
    axes[5].set_title(f"{title_prefix}Generalization gap")
    axes[5].set_xlabel("Epoch")
    axes[5].set_ylabel("Gap (positive = overfitting)")
    axes[5].legend(fontsize=8)
    axes[5].grid(alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_lr_curve(history, out_path, title="Learning rate schedule"):
    """Standalone learning-rate plot."""
    lrs = history.get("learning_rate") or []
    fig, ax = plt.subplots(figsize=(6, 4))
    if lrs:
        ax.plot(range(1, len(lrs) + 1), lrs, marker="d", color="tab:green")
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "Learning rate not recorded for this run",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate (log scale)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def make_overlay(image_01, mask_bin, color=(1.0, 0.0, 0.0), alpha=0.4):
    """image_01: float [0,1] grayscale HxW. mask_bin: {0,1} HxW. Returns HxWx3 RGB float [0,1]."""
    rgb = np.stack([image_01] * 3, axis=-1)
    overlay = rgb.copy()
    for c in range(3):
        overlay[..., c] = np.where(mask_bin > 0, (1 - alpha) * rgb[..., c] + alpha * color[c], rgb[..., c])
    return overlay


def plot_sample_panel(image_01, gt_mask, pred_mask, out_path, title=""):
    """5-panel: original, GT mask, predicted mask, overlay, GT-vs-pred comparison."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    axes[0].imshow(image_01, cmap="gray")
    axes[0].set_title("Original CT")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground truth mask")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Predicted mask")
    axes[2].axis("off")

    overlay = make_overlay(image_01, pred_mask, color=(1, 0, 0), alpha=0.45)
    axes[3].imshow(overlay)
    axes[3].set_title("Prediction overlay")
    axes[3].axis("off")

    # comparison: TP=white, FP=red, FN=blue, TN=black
    comp = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
    tp = (pred_mask == 1) & (gt_mask == 1)
    fp = (pred_mask == 1) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask == 1)
    comp[tp] = [1, 1, 1]
    comp[fp] = [1, 0, 0]
    comp[fn] = [0, 0.4, 1]
    axes[4].imshow(comp)
    axes[4].set_title("GT vs Pred\n(white=TP, red=FP, blue=FN)")
    axes[4].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_error_map(image_01, gt_mask, pred_mask, out_path, title=""):
    """Single-panel |pred - gt| error heatmap over the image."""
    error = np.abs(pred_mask.astype(np.float32) - gt_mask.astype(np.float32))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_01, cmap="gray")
    im = ax.imshow(error, cmap="hot", alpha=0.6)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_probability_map(image_01, prob_map, out_path, title="Prediction probability map"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_01, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")
    im = axes[1].imshow(prob_map, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_full_panel(image_01, gt_mask, pred_mask, prob_map, out_path, title=""):
    """
    6-panel figure: original / ground truth / prediction / overlay /
    GT-vs-pred error map / probability map.

    This is the figure used for the result gallery, because it shows the
    model's CONFIDENCE alongside its thresholded output — a prediction that is
    wrong but uncertain (probabilities near 0.5) is a different failure from
    one that is wrong and confident.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    axes[0].imshow(image_01, cmap="gray")
    axes[0].set_title("1. Original CT slice")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"2. Ground truth ({int(gt_mask.sum())} px)")

    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"3. Prediction ({int(pred_mask.sum())} px)")

    ov = make_overlay(image_01, gt_mask, color=(0, 1, 0), alpha=0.40)
    ov = make_overlay_on_rgb(ov, pred_mask, color=(1, 0, 0), alpha=0.35)
    axes[3].imshow(ov)
    axes[3].set_title("4. Overlay (green=GT, red=prediction)")

    comp = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
    comp[(pred_mask == 1) & (gt_mask == 1)] = [1, 1, 1]
    comp[(pred_mask == 1) & (gt_mask == 0)] = [1, 0, 0]
    comp[(pred_mask == 0) & (gt_mask == 1)] = [0, 0.4, 1]
    axes[4].imshow(comp)
    axes[4].set_title("5. Error map\n(white=TP, red=FP, blue=FN)")

    im = axes[5].imshow(prob_map, cmap="viridis", vmin=0, vmax=1)
    axes[5].set_title("6. Predicted probability map")
    plt.colorbar(im, ax=axes[5], fraction=0.046)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_overlay_on_rgb(rgb, mask_bin, color=(1.0, 0.0, 0.0), alpha=0.4):
    """Blend a second mask onto an already-RGB overlay image."""
    out = rgb.copy()
    for c in range(3):
        out[..., c] = np.where(mask_bin > 0,
                               (1 - alpha) * out[..., c] + alpha * color[c],
                               out[..., c])
    return out


def plot_gallery(entries, out_path, title="Result gallery", cols=4):
    """
    Contact-sheet gallery of many cases at once.

    `entries` is a list of dicts with keys: image (HxW float [0,1]), gt, pred,
    label. Each cell shows the GT/pred error map over the CT image so success
    and failure are directly comparable across the sheet. This is the figure
    that demonstrates the model was not evaluated on cherry-picked successes.
    """
    n = len(entries)
    if n == 0:
        return
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.7 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, e in zip(axes, entries):
        img, gt, pred = e["image"], e["gt"], e["pred"]
        base = np.stack([img] * 3, axis=-1)
        base = make_overlay_on_rgb(base, (pred == 1) & (gt == 1), (1, 1, 1), 0.55)
        base = make_overlay_on_rgb(base, (pred == 1) & (gt == 0), (1, 0, 0), 0.55)
        base = make_overlay_on_rgb(base, (pred == 0) & (gt == 1), (0, 0.4, 1), 0.55)
        ax.imshow(np.clip(base, 0, 1))
        ax.set_title(e.get("label", ""), fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"{title}\nwhite = correct foreground, red = false positive, "
                 f"blue = missed foreground", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
