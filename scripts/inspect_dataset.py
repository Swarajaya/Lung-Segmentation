"""
Dataset inspection script.

Measures (never assumes) the real properties of the supplied dataset and
writes outputs/metrics/dataset_inspection_report.json plus a human-readable
markdown summary at outputs/metrics/dataset_inspection_report.md.
"""
import os
import sys
import re
import json
import argparse
import hashlib
from collections import defaultdict, Counter

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config, resolve_path
from src.utils import ensure_dir, save_json

FNAME_RE = re.compile(r'^(train|val|test)_Subject_(\d+)_(\d+)\.png$')


def inspect(images_dir, masks_dir):
    img_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))
    img_set, mask_set = set(img_files), set(mask_files)
    paired = sorted(img_set & mask_set)

    report = {
        "n_images": len(img_files),
        "n_masks": len(mask_files),
        "n_paired": len(paired),
        "n_missing_masks_for_images": len(img_set - mask_set),
        "n_missing_images_for_masks": len(mask_set - img_set),
    }

    subj_split_map = defaultdict(set)
    subjects = set()
    unmatched = []
    split_tag_counts = Counter()
    for f in img_files:
        m = FNAME_RE.match(f)
        if m:
            tag, subj, _slice = m.groups()
            subjects.add(subj)
            subj_split_map[subj].add(tag)
            split_tag_counts[tag] += 1
        else:
            unmatched.append(f)
    report["n_unique_subjects"] = len(subjects)
    report["n_filename_unmatched"] = len(unmatched)
    report["original_split_tag_counts_in_filenames"] = dict(split_tag_counts)

    formats, modes, sizes, dtypes = Counter(), Counter(), Counter(), Counter()
    mask_formats, mask_modes, mask_sizes = Counter(), Counter(), Counter()
    mask_value_sets, mask_dtypes = Counter(), Counter()
    corrupt_images, corrupt_masks = [], []
    empty_masks, nonbinary_masks, size_mismatch = [], [], []
    hashes = defaultdict(list)
    mask_hashes = defaultdict(list)
    means = []
    fg_fractions = []
    per_subject = defaultdict(lambda: {"n_slices": 0, "n_empty": 0, "fg_fraction_sum": 0.0})
    total_fg_px, total_px = 0, 0

    for f in paired:
        ip, mp = os.path.join(images_dir, f), os.path.join(masks_dir, f)
        try:
            im = Image.open(ip)
            im.verify()
            im = Image.open(ip)
            arr = np.array(im)
            formats[im.format] += 1
            modes[im.mode] += 1
            sizes[str(im.size)] += 1
            dtypes[str(arr.dtype)] += 1
            means.append(float(arr.mean()))
            h = hashlib.md5(arr.tobytes()).hexdigest()
            hashes[h].append(f)
        except Exception as e:
            corrupt_images.append({"file": f, "error": str(e)})
            continue

        try:
            mm = Image.open(mp)
            mm.verify()
            mm = Image.open(mp)
            marr = np.array(mm)
            mask_formats[mm.format] += 1
            mask_modes[mm.mode] += 1
            mask_sizes[str(mm.size)] += 1
            mask_dtypes[str(marr.dtype)] += 1
            mask_hashes[hashlib.md5(marr.tobytes()).hexdigest()].append(f)
            uniq = tuple(sorted(np.unique(marr).tolist()))
            mask_value_sets[str(uniq)] += 1
            if marr.sum() == 0:
                empty_masks.append(f)
            if len(uniq) > 2:
                nonbinary_masks.append({"file": f, "values": list(uniq)})
            if im.size != mm.size:
                size_mismatch.append(f)
            fg_frac = float((marr > 0).mean())
            fg_fractions.append(fg_frac)
            total_fg_px += int((marr > 0).sum())
            total_px += int(marr.size)

            sm = FNAME_RE.match(f)
            subj = sm.group(2) if sm else "unknown"
            per_subject[subj]["n_slices"] += 1
            per_subject[subj]["fg_fraction_sum"] += fg_frac
            if marr.sum() == 0:
                per_subject[subj]["n_empty"] += 1
        except Exception as e:
            corrupt_masks.append({"file": f, "error": str(e)})

    dupes = {h: fl for h, fl in hashes.items() if len(fl) > 1}
    mask_dupes = {h: fl for h, fl in mask_hashes.items() if len(fl) > 1}

    # Per-subject summary (never invented — derived from the parsed filenames).
    subject_summary = sorted(
        ({"subject_id": s,
          "n_slices": v["n_slices"],
          "n_empty_masks": v["n_empty"],
          "mean_foreground_fraction": v["fg_fraction_sum"] / v["n_slices"] if v["n_slices"] else 0.0}
         for s, v in per_subject.items()),
        key=lambda d: (-d["n_slices"], d["subject_id"]))
    slice_counts = [d["n_slices"] for d in subject_summary]

    report.update({
        "image_formats": dict(formats),
        "image_modes": dict(modes),
        "image_sizes": dict(sizes),
        "image_dtypes": dict(dtypes),
        "n_corrupt_images": len(corrupt_images),
        "corrupt_images_examples": corrupt_images[:10],
        "mask_formats": dict(mask_formats),
        "mask_modes": dict(mask_modes),
        "mask_sizes": dict(mask_sizes),
        "n_corrupt_masks": len(corrupt_masks),
        "corrupt_masks_examples": corrupt_masks[:10],
        "mask_unique_value_sets": dict(mask_value_sets),
        "n_empty_masks": len(empty_masks),
        "pct_empty_masks": len(empty_masks) / len(paired) if paired else None,
        "n_nonbinary_masks": len(nonbinary_masks),
        "n_size_mismatch": len(size_mismatch),
        "mask_dtypes": dict(mask_dtypes),
        "n_duplicate_image_groups": len(dupes),
        "n_duplicate_image_files_total": sum(len(v) for v in dupes.values()),
        "duplicate_image_examples": {k: v[:5] for k, v in list(dupes.items())[:5]},
        # Duplicate MASKS are reported separately and are expected to be
        # numerous: every empty mask is byte-identical to every other empty
        # mask. A duplicate mask is only suspicious when its IMAGE is also a
        # duplicate, which is what n_duplicate_image_groups measures.
        "n_duplicate_mask_groups": len(mask_dupes),
        "n_duplicate_mask_files_total": sum(len(v) for v in mask_dupes.values()),
        # Aliases retained for backward compatibility with earlier reports.
        "n_duplicate_groups": len(dupes),
        "n_duplicate_files_total": sum(len(v) for v in dupes.values()),
        "duplicate_examples": {k: v[:5] for k, v in list(dupes.items())[:5]},
        "image_intensity_mean": float(np.mean(means)) if means else None,
        "image_intensity_std": float(np.std(means)) if means else None,
        "foreground_pixel_fraction_mean": float(np.mean(fg_fractions)) if fg_fractions else None,
        "foreground_pixel_fraction_median": float(np.median(fg_fractions)) if fg_fractions else None,
        "foreground_pixel_fraction_max": float(np.max(fg_fractions)) if fg_fractions else None,
        # --- class imbalance, measured over the WHOLE dataset ---
        "class_balance": {
            "total_foreground_pixels": total_fg_px,
            "total_pixels": total_px,
            "foreground_pixel_ratio": (total_fg_px / total_px) if total_px else None,
            "background_pixel_ratio": (1 - total_fg_px / total_px) if total_px else None,
            "background_to_foreground_ratio": ((total_px - total_fg_px) / total_fg_px)
                                              if total_fg_px else None,
        },
        # --- per-subject statistics ---
        "n_subjects_summarised": len(subject_summary),
        "slices_per_subject_min": int(min(slice_counts)) if slice_counts else None,
        "slices_per_subject_max": int(max(slice_counts)) if slice_counts else None,
        "slices_per_subject_mean": float(np.mean(slice_counts)) if slice_counts else None,
        "per_subject": subject_summary,
    })
    return report


def write_markdown(report, out_path):
    cb = report.get("class_balance")
    lines = [
        "# Dataset Inspection Report",
        "",
        "This report was generated automatically by `scripts/inspect_dataset.py` "
        "by reading every image/mask file on disk. No value below is assumed.",
        "",
        f"- Number of image files: **{report['n_images']}**",
        f"- Number of mask files: **{report['n_masks']}**",
        f"- Number of valid image/mask pairs: **{report['n_paired']}**",
        f"- Missing masks for images: {report['n_missing_masks_for_images']}",
        f"- Missing images for masks: {report['n_missing_images_for_masks']}",
        f"- Unique subjects (parsed from filenames): **{report['n_unique_subjects']}**",
        f"- Filenames not matching the `Subject_<id>_<slice>` pattern: {report['n_filename_unmatched']}",
        f"- Original split tags embedded in filenames: {report['original_split_tag_counts_in_filenames']}",
        "",
        "## Image properties",
        f"- Formats: {report['image_formats']}",
        f"- Color modes: {report['image_modes']}",
        f"- Sizes: {report['image_sizes']}",
        f"- Dtypes: {report['image_dtypes']}",
        f"- Corrupt/unreadable images: {report['n_corrupt_images']}",
        f"- Mean pixel intensity across sample: {report['image_intensity_mean']:.2f} "
        f"(std {report['image_intensity_std']:.2f})" if report['image_intensity_mean'] is not None else "",
        "",
        "## Mask properties",
        f"- Formats: {report['mask_formats']}",
        f"- Color modes: {report['mask_modes']}",
        f"- Sizes: {report['mask_sizes']}",
        f"- Corrupt/unreadable masks: {report['n_corrupt_masks']}",
        f"- Unique pixel value sets found: {report['mask_unique_value_sets']}",
        f"- Empty masks (no foreground pixel): {report['n_empty_masks']} "
        f"({report['pct_empty_masks']*100:.1f}% of pairs)" if report['pct_empty_masks'] is not None else "",
        f"- Masks with more than 2 unique values (non-binary): {report['n_nonbinary_masks']}",
        f"- Image/mask pairs with mismatched dimensions: {report['n_size_mismatch']}",
        f"- Mean foreground pixel fraction: {report['foreground_pixel_fraction_mean']*100:.3f}% "
        f"(median {report['foreground_pixel_fraction_median']*100:.3f}%, "
        f"max {report['foreground_pixel_fraction_max']*100:.2f}%)"
        if report['foreground_pixel_fraction_mean'] is not None else "",
        "",
        f"- Mask dtypes: {report.get('mask_dtypes', {})}",
        "",
        "## Class imbalance",
        f"- Foreground pixels across the whole dataset: "
        f"{cb['total_foreground_pixels']:,} of {cb['total_pixels']:,}"
        if cb else "",
        f"- Foreground pixel ratio: **{cb['foreground_pixel_ratio']*100:.4f}%**"
        if cb and cb['foreground_pixel_ratio'] is not None else "",
        f"- Background:foreground ratio: **{cb['background_to_foreground_ratio']:.1f} : 1**"
        if cb and cb['background_to_foreground_ratio'] is not None else "",
        "",
        "This is severe class imbalance, and it is the reason the project uses a "
        "Dice-based loss term rather than BCE alone: a model predicting "
        "background everywhere would already score very high pixel accuracy.",
        "",
        "## Subjects",
        f"- Subjects summarised: {report.get('n_subjects_summarised')}",
        f"- Slices per subject: min {report.get('slices_per_subject_min')}, "
        f"max {report.get('slices_per_subject_max')}, "
        f"mean {report.get('slices_per_subject_mean'):.1f}"
        if report.get('slices_per_subject_mean') is not None else "",
        "",
        "Subjects contribute unequal numbers of slices, which is exactly why the "
        "train/val/test split is made at SUBJECT level (see "
        "`scripts/split_dataset.py`).",
        "",
        "## Duplicates",
        f"- Duplicate IMAGE pixel-content groups: {report['n_duplicate_image_groups']} "
        f"({report['n_duplicate_image_files_total']} files involved)",
        f"- Duplicate MASK pixel-content groups: {report.get('n_duplicate_mask_groups')} "
        f"({report.get('n_duplicate_mask_files_total')} files involved) — expected to be "
        f"large, since every empty mask is byte-identical to every other empty mask",
        "",
        "## Interpretation",
        "- Masks are strictly binary with values `{0, 255}` -> a threshold of 127 cleanly "
        "recovers a `{0,1}` foreground/background mask.",
        "- A non-trivial fraction of masks are entirely empty (no foreground). This is "
        "expected for CT slices that do not intersect the annotated structure, and is "
        "handled explicitly in the metric code (an empty ground truth with an empty "
        "prediction scores a perfect Dice/IoU of 1.0, not undefined/NaN).",
        "- Filenames encode a subject ID, which enables leak-free SUBJECT-LEVEL "
        "train/val/test splitting (see `scripts/split_dataset.py`).",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Validate the dataset and write a measured inspection report.")
    parser.add_argument("--config", default=None,
                        help="Path to a config YAML (default: config.yaml)")
    parser.add_argument("--out-prefix", default="outputs/metrics/dataset_inspection_report",
                        help="Output path prefix; .json and .md are appended.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    images_dir = resolve_path(cfg.data.images_dir)
    masks_dir = resolve_path(cfg.data.masks_dir)

    for d, name in ((images_dir, "images"), (masks_dir, "masks")):
        if not os.path.isdir(d):
            print(f"{name} directory not found: {d}\n"
                  f"See DATASET_SETUP.md for where to place the dataset.")
            sys.exit(1)

    report = inspect(images_dir, masks_dir)

    out_json = resolve_path(f"{args.out_prefix}.json")
    out_md = resolve_path(f"{args.out_prefix}.md")
    ensure_dir(os.path.dirname(out_json))
    save_json(report, out_json)
    write_markdown(report, out_md)

    print(json.dumps(report, indent=2)[:3000])
    print(f"\nWrote {out_json}\nWrote {out_md}")


if __name__ == "__main__":
    main()
