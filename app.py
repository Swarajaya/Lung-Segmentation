"""
Streamlit demo for the Lung CT Segmentation U-Net project.

Run with:  streamlit run app/app.py

Handles gracefully: no trained checkpoint, no dataset present, non-square or
non-grayscale uploads. Every number shown is computed from the actual model
output — nothing is illustrative.
"""
import os
import sys
import io
import glob

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.config import load_config, resolve_path
from src.predict import load_model_for_inference, predict_from_config, mask_statistics
from src.preprocessing import load_mask, preprocess_mask
from src.metrics import compute_all_metrics

st.set_page_config(page_title="Lung CT Segmentation — Research Demo", layout="wide")

DISCLAIMER = (
    "**Medical disclaimer:** This is a research/educational prototype and is "
    "not intended for clinical diagnosis or medical decision-making."
)


@st.cache_resource
def get_config():
    return load_config()


@st.cache_resource
def get_model(ckpt_path):
    return load_model_for_inference(ckpt_path, get_config())


def find_checkpoints():
    d = resolve_path("outputs/checkpoints")
    if not os.path.isdir(d):
        return []
    return sorted(glob.glob(os.path.join(d, "*_best.pt")))


def colorize(prob_map):
    """
    Map a [0,1] probability array to an RGB heatmap without importing pyplot.

    Uses `matplotlib.colormaps`, the supported API. The older
    `matplotlib.cm.get_cmap` was deprecated in 3.7 and is removed in 3.11, so
    the fallback keeps the app working on both old and new installs.
    """
    try:
        import matplotlib
        cmap = matplotlib.colormaps["viridis"]
    except (ImportError, AttributeError, KeyError):
        import matplotlib.cm as cm
        cmap = cm.get_cmap("viridis")
    return (cmap(np.clip(prob_map, 0, 1))[..., :3]).astype(np.float32)


def overlay_rgb(gray01, mask, color, alpha=0.45, base=None):
    rgb = base if base is not None else np.stack([gray01] * 3, axis=-1)
    out = rgb.copy()
    for c in range(3):
        out[..., c] = np.where(mask > 0, (1 - alpha) * out[..., c] + alpha * color[c], out[..., c])
    return np.clip(out, 0, 1)


def sidebar(cfg, checkpoints):
    with st.sidebar:
        st.header("Model")
        ckpt = st.selectbox("Checkpoint", checkpoints, format_func=os.path.basename)
        threshold = st.slider(
            "Segmentation threshold", 0.05, 0.95, float(cfg.evaluation.threshold), 0.05,
            help="Probability above which a pixel is labelled foreground. Lowering it "
                 "increases recall and false positives; raising it does the opposite.")
        st.markdown("---")
        st.caption(
            f"Input size: {cfg.data.image_size}×{cfg.data.image_size}  \n"
            f"Architecture: {cfg.model.name} (base_channels={cfg.model.base_channels})  \n"
            f"Normalization: {cfg.data.get('normalization', 'zscore_dataset')}  \n"
            f"CLAHE: {'on' if cfg.data.get('use_clahe') else 'off'}")
        st.markdown("---")
        st.caption("Normalization statistics are read from the checkpoint, so the app "
                   "preprocesses exactly as the model was trained.")
    return ckpt, threshold


def main():
    cfg = get_config()

    st.title("Lung CT Segmentation — Research Demo")
    st.caption("U-Net semantic segmentation of lung CT slices.")
    st.warning(DISCLAIMER, icon="⚠️")

    checkpoints = find_checkpoints()
    if not checkpoints:
        st.error(
            "No trained checkpoint found in `outputs/checkpoints/`.\n\n"
            "Train one first:\n\n"
            "```bash\n"
            "python scripts/split_dataset.py\n"
            "python scripts/train.py --mode smoke_test\n"
            "```\n\n"
            "Use `--mode full` on a GPU machine for a real model, then reload this page.")
        st.stop()

    ckpt_choice, threshold = sidebar(cfg, checkpoints)
    model, norm_stats, device = get_model(ckpt_choice)

    tab_upload, tab_samples = st.tabs(["Upload an image", "Try a dataset sample"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a lung CT slice (PNG / JPG). Colour images are converted to grayscale.",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
        if uploaded is not None:
            image = Image.open(uploaded).convert("L")
            st.subheader("Original image")
            st.image(image, caption=f"{uploaded.name} — {image.size[0]}×{image.size[1]} px",
                     width=320)
            if st.button("Run segmentation", type="primary", key="run_upload"):
                run_and_display(model, norm_stats, device, image, cfg, threshold,
                                name=os.path.splitext(uploaded.name)[0], gt_filename=uploaded.name)
        else:
            st.info("Upload an image to run segmentation, or switch to the "
                    "**Try a dataset sample** tab.")

    with tab_samples:
        sample_dir = resolve_path(cfg.data.images_dir)
        if not os.path.isdir(sample_dir):
            st.info("The dataset is not present in this copy of the project. "
                    "See `DATASET_SETUP.md` for where to place it, or use the "
                    "**Upload an image** tab.")
        else:
            names = sorted(os.listdir(sample_dir))
            pick = st.selectbox("Pick a sample from the dataset", names[:500])
            cols = st.columns(6)
            for col, fname in zip(cols, names[:6]):
                with col:
                    st.image(Image.open(os.path.join(sample_dir, fname)).convert("L"),
                             caption=fname, use_container_width=True)
            if st.button("Run segmentation", type="primary", key="run_sample"):
                image = Image.open(os.path.join(sample_dir, pick)).convert("L")
                run_and_display(model, norm_stats, device, image, cfg, threshold,
                                name=os.path.splitext(pick)[0], gt_filename=pick)


def run_and_display(model, norm_stats, device, image, cfg, threshold, name, gt_filename=None):
    img_gray = np.array(image)
    prob_map, pred_mask, resized = predict_from_config(
        model, norm_stats, device, img_gray, cfg, threshold=threshold)
    resized01 = resized.astype(np.float32) / 255.0
    stats = mask_statistics(pred_mask, prob_map)

    st.subheader("Segmentation result")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(resized01, caption="Model input (resized)", use_container_width=True, clamp=True)
    with c2:
        st.image(pred_mask.astype(np.float32), caption="Predicted mask",
                 use_container_width=True, clamp=True)
    with c3:
        st.image(overlay_rgb(resized01, pred_mask, (1, 0, 0)),
                 caption="Overlay (prediction in red)", use_container_width=True, clamp=True)
    with c4:
        st.image(colorize(prob_map), caption="Probability map (sigmoid output)",
                 use_container_width=True, clamp=True)

    st.subheader("Segmentation statistics")
    m = st.columns(5)
    m[0].metric("Image size", f"{stats['width']}×{stats['height']}")
    m[1].metric("Foreground pixels", f"{stats['foreground_pixels']:,}")
    m[2].metric("Foreground share", f"{stats['foreground_percent']:.3f}%")
    m[3].metric("Connected components", f"{stats['n_components']}"
                if stats["n_components"] is not None else "n/a")
    m[4].metric("Mean confidence (fg)",
                f"{stats['mean_foreground_confidence']:.3f}"
                if stats["mean_foreground_confidence"] is not None else "n/a")

    bb = stats["bounding_box"]
    if bb:
        st.caption(f"Bounding box of predicted region: x [{bb['x_min']}, {bb['x_max']}], "
                   f"y [{bb['y_min']}, {bb['y_max']}] — {bb['width']}×{bb['height']} px.")
    else:
        st.caption("No foreground predicted at this threshold, so no bounding box is "
                   "defined. Lowering the threshold in the sidebar will make the model "
                   "more permissive.")

    st.caption("Areas are reported in **pixels**. This dataset carries no pixel-spacing "
               "(mm/pixel) metadata, so physical area or volume cannot be derived from it.")

    # ---- ground truth comparison, only when a real mask exists ----
    if gt_filename:
        gt_path = os.path.join(resolve_path(cfg.data.masks_dir), gt_filename)
        if os.path.exists(gt_path):
            gt_bin = preprocess_mask(load_mask(gt_path), cfg.data.image_size,
                                     cfg.data.mask_threshold)
            met = compute_all_metrics(pred_mask, gt_bin, include_surface=True)
            st.subheader("Comparison against ground truth")
            g = st.columns(5)
            g[0].metric("Dice", f"{met['dice']:.4f}")
            g[1].metric("IoU", f"{met['iou']:.4f}")
            g[2].metric("Precision", f"{met['precision']:.4f}")
            g[3].metric("Recall", f"{met['recall']:.4f}")
            g[4].metric("HD95 (px)", f"{met['hd95']:.2f}" if met["hd95"] is not None else "n/a")

            comp = np.zeros((*gt_bin.shape, 3), dtype=np.float32)
            comp[(pred_mask == 1) & (gt_bin == 1)] = [1, 1, 1]
            comp[(pred_mask == 1) & (gt_bin == 0)] = [1, 0, 0]
            comp[(pred_mask == 0) & (gt_bin == 1)] = [0, 0.4, 1]
            e1, e2 = st.columns(2)
            with e1:
                st.image(overlay_rgb(resized01, gt_bin, (0, 1, 0)),
                         caption="Ground truth (green)", use_container_width=True, clamp=True)
            with e2:
                st.image(comp, caption="Error map — white = correct, red = false positive, "
                                       "blue = missed", use_container_width=True, clamp=True)
            if met["gt_foreground_fraction"] == 0:
                st.info("This slice has an EMPTY ground-truth mask. Dice is 1.0 only if "
                        "the model also predicts nothing; any predicted pixel here is a "
                        "false positive.")

    buf = io.BytesIO()
    Image.fromarray((pred_mask * 255).astype(np.uint8)).save(buf, format="PNG")
    st.download_button("Download predicted mask (PNG)", buf.getvalue(),
                       file_name=f"{name}_mask.png", mime="image/png")

    st.markdown("---")
    st.caption(DISCLAIMER + " The model has not been validated on any clinical "
               "population, and no claim of clinical accuracy is made.")


if __name__ == "__main__":
    main()
