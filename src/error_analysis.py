"""
Automated error analysis.

The point of this module is to make the project critique its own model rather
than report a single flattering average. It takes the per-sample metrics
produced by `src.evaluate.evaluate_checkpoint` and sorts every test image into
named, explicitly-defined failure categories, then summarises how often each
occurs and which images are worst.

Failure categories (thresholds configurable in config.yaml -> error_analysis):

  low_dice              Dice below `low_dice_threshold` on an image whose
                        ground truth is NON-empty. Empty-GT images are
                        excluded from this category because their Dice is 1.0
                        or 0.0 by definition and would distort the count.
  low_iou               Same, for IoU below `low_iou_threshold`.
  high_false_positive   More than `high_fp_pixel_threshold` FP pixels: the
                        model is over-segmenting, labelling background as
                        lesion/lung. Clinically this is a false alarm.
  high_false_negative   More than `high_fn_pixel_threshold` FN pixels: the
                        model is under-segmenting and MISSING real structure.
                        In a screening context this is the more dangerous of
                        the two errors, which is why the two are counted
                        separately instead of being merged into one "error"
                        number.
  missed_positive       Ground truth has foreground but the model predicted a
                        COMPLETELY empty mask — a total miss, the worst
                        individual failure mode.
  false_alarm_on_empty  Ground truth is empty but the model predicted
                        foreground anyway — hallucinated structure on a slice
                        that contains none.
  correct_empty         Ground truth empty and prediction empty. Not a
                        failure; tracked so the empty-slice behaviour is fully
                        accounted for.

A single image may fall into several categories (e.g. both high FP and low
Dice); the counts are therefore not mutually exclusive and this is stated in
the generated report.
"""
import os
import csv


def categorize_sample(m: dict, cfg_ea) -> list:
    """Return the list of failure-category labels that apply to one sample."""
    gt_positive = (m["tp"] + m["fn"]) > 0
    pred_positive = (m["tp"] + m["fp"]) > 0
    tags = []

    if gt_positive and not pred_positive:
        tags.append("missed_positive")
    if (not gt_positive) and pred_positive:
        tags.append("false_alarm_on_empty")
    if (not gt_positive) and (not pred_positive):
        tags.append("correct_empty")

    if gt_positive:
        if m["dice"] < cfg_ea.low_dice_threshold:
            tags.append("low_dice")
        if m["iou"] < cfg_ea.low_iou_threshold:
            tags.append("low_iou")

    if m["fp"] > cfg_ea.high_fp_pixel_threshold:
        tags.append("high_false_positive")
    if m["fn"] > cfg_ea.high_fn_pixel_threshold:
        tags.append("high_false_negative")

    return tags


def analyze(per_sample_metrics: list, cfg) -> dict:
    """Categorize every sample and build a summary dict."""
    ea = cfg.error_analysis
    categories = {}
    tagged = []

    for m in per_sample_metrics:
        tags = categorize_sample(m, ea)
        record = {
            "filename": m.get("filename"),
            "subject_id": m.get("subject_id"),
            "dice": m["dice"], "iou": m["iou"],
            "precision": m["precision"], "recall": m["recall"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "gt_area_px": m["tp"] + m["fn"],
            "pred_area_px": m["tp"] + m["fp"],
            "hd95": m.get("hd95"), "assd": m.get("assd"),
            "categories": tags,
        }
        tagged.append(record)
        for t in tags:
            categories.setdefault(t, []).append(record["filename"])

    gt_positive = [r for r in tagged if r["gt_area_px"] > 0]
    worst = sorted(gt_positive, key=lambda r: r["dice"])[:ea.n_worst_to_report]

    # Subject-level view: is the model failing on specific patients rather
    # than uniformly? This distinguishes "hard subject" from "hard slice".
    by_subject = {}
    for r in gt_positive:
        by_subject.setdefault(r["subject_id"] or "unknown", []).append(r["dice"])
    subject_mean_dice = sorted(
        ({"subject_id": s, "n_positive_slices": len(v),
          "mean_dice": sum(v) / len(v)} for s, v in by_subject.items()),
        key=lambda d: d["mean_dice"])

    total_fp = sum(r["fp"] for r in tagged)
    total_fn = sum(r["fn"] for r in tagged)

    return {
        "n_samples": len(tagged),
        "n_gt_positive": len(gt_positive),
        "n_gt_empty": len(tagged) - len(gt_positive),
        "thresholds_used": {
            "low_dice_threshold": ea.low_dice_threshold,
            "low_iou_threshold": ea.low_iou_threshold,
            "high_fp_pixel_threshold": ea.high_fp_pixel_threshold,
            "high_fn_pixel_threshold": ea.high_fn_pixel_threshold,
        },
        "category_counts": {k: len(v) for k, v in sorted(categories.items())},
        "category_examples": {k: v[:10] for k, v in sorted(categories.items())},
        "total_false_positive_px": total_fp,
        "total_false_negative_px": total_fn,
        "fp_fn_ratio": (total_fp / total_fn) if total_fn else None,
        "worst_cases_by_dice": worst,
        "subject_mean_dice_ranked": subject_mean_dice,
        "per_sample": tagged,
    }


def write_csv(analysis: dict, out_path: str):
    """Write the per-sample categorization to CSV."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = ["filename", "subject_id", "dice", "iou", "precision", "recall",
              "tp", "fp", "fn", "gt_area_px", "pred_area_px", "hd95", "assd",
              "categories"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in analysis["per_sample"]:
            row = dict(r)
            row["categories"] = ";".join(r["categories"])
            w.writerow({k: row.get(k, "") for k in fields})


def write_markdown(analysis: dict, out_path: str, experiment_name: str, mode: str):
    """Write the human-readable error-analysis report."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    a = analysis
    L = [
        f"# Error Analysis — `{experiment_name}` ({mode})",
        "",
        "Generated automatically by `scripts/error_analysis.py` from the actual "
        "per-sample evaluation output. Every number below is measured.",
        "",
        f"- Images analysed: **{a['n_samples']}**",
        f"- With non-empty ground truth: **{a['n_gt_positive']}**",
        f"- With empty ground truth: **{a['n_gt_empty']}**",
        "",
        "## Failure categories",
        "",
        "Categories are **not mutually exclusive** — one image can be both "
        "`low_dice` and `high_false_positive`.",
        "",
        "| Category | Count |",
        "|---|---|",
    ]
    for k, v in a["category_counts"].items():
        L.append(f"| `{k}` | {v} |")

    L += [
        "",
        f"Thresholds used: {a['thresholds_used']}",
        "",
        "## False positive vs false negative balance",
        "",
        f"- Total false-positive pixels: **{a['total_false_positive_px']:,}**",
        f"- Total false-negative pixels: **{a['total_false_negative_px']:,}**",
        f"- FP/FN ratio: **{a['fp_fn_ratio']:.3f}**" if a["fp_fn_ratio"] is not None
        else "- FP/FN ratio: undefined (no false negatives)",
        "",
        "A ratio far above 1 means the model over-segments (false alarms); far "
        "below 1 means it under-segments (missed structure). The two errors are "
        "not clinically equivalent, so they are reported separately rather than "
        "collapsed into one figure.",
        "",
        "## Worst cases by Dice (non-empty ground truth only)",
        "",
        "| File | Subject | Dice | IoU | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in a["worst_cases_by_dice"]:
        L.append(f"| `{r['filename']}` | {r['subject_id']} | {r['dice']:.4f} | "
                 f"{r['iou']:.4f} | {r['tp']} | {r['fp']} | {r['fn']} |")

    L += [
        "",
        "## Per-subject mean Dice (worst 10, non-empty slices only)",
        "",
        "If a few subjects dominate the bottom of this table, the model is "
        "failing on particular patients (anatomy/protocol) rather than "
        "uniformly across slices.",
        "",
        "| Subject | Positive slices | Mean Dice |",
        "|---|---|---|",
    ]
    for s in a["subject_mean_dice_ranked"][:10]:
        L.append(f"| {s['subject_id']} | {s['n_positive_slices']} | {s['mean_dice']:.4f} |")

    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
