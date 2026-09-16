"""
Subject-level dataset splitting and leakage verification.

Why subject-level and not random image-level splitting
------------------------------------------------------
Filenames in this dataset follow `<origtag>_Subject_<id>_<slice>.png`: many CT
slices belong to the SAME patient. Adjacent slices of one patient are highly
correlated — same anatomy, same scanner, same reconstruction, often nearly the
same lesion a few millimetres up or down. If slices are split randomly, slices
from one patient land in both the training and the test set, and the model can
score well on test simply by recognising anatomy it memorised during training.
The reported metric then measures memorisation, not generalisation to a NEW
patient, which is the only thing that matters clinically.

Splitting by subject removes that path entirely: every slice from a given
subject goes into exactly one of train/val/test.

The original filenames also contain a "train"/"val" tag from whatever source
the files were distributed from. We deliberately do NOT trust it: it defines
no test set, and we cannot verify it was built subject-wise. We derive our own
split and document that decision rather than inheriting an unverifiable one.
"""
import os
import re
import random

SUBJECT_FNAME_RE = re.compile(r'^(?:train|val|test)_Subject_(\d+)_(\d+)\.png$', re.IGNORECASE)
FALLBACK_SUBJECT_RE = re.compile(r'Subject[_-](\d+)', re.IGNORECASE)


def parse_subject(fname: str):
    """Subject ID for a filename, or None when no subject is encoded."""
    m = SUBJECT_FNAME_RE.match(fname)
    if m:
        return m.group(1)
    m = FALLBACK_SUBJECT_RE.search(fname)
    return m.group(1) if m else None


def group_by_subject(files):
    """Map subject ID -> list of filenames. Unparsable names returned separately."""
    subject_to_files, unmatched = {}, []
    for f in files:
        subj = parse_subject(f)
        if subj is None:
            unmatched.append(f)
        else:
            subject_to_files.setdefault(subj, []).append(f)
    return subject_to_files, unmatched


def build_split(images_dir_or_files, val_fraction, test_fraction, seed,
                split_level="subject"):
    """
    Build a reproducible split.

    `split_level="subject"` (the default and the only correct option for this
    dataset) partitions SUBJECTS. `split_level="image"` is supported only as a
    deliberate, documented fallback for datasets with no recoverable subject
    ID — it is NOT used here, and the resulting split file records which mode
    produced it so a leaked split can never be mistaken for a clean one.
    """
    if isinstance(images_dir_or_files, str):
        files = sorted(f for f in os.listdir(images_dir_or_files) if f.lower().endswith(".png"))
    else:
        files = sorted(images_dir_or_files)

    subject_to_files, unmatched = group_by_subject(files)

    if split_level == "subject" and not subject_to_files:
        raise ValueError(
            "split_level='subject' requested but no subject IDs could be parsed "
            "from any filename. Either fix the filenames or set "
            "data.split_level: 'image' in config.yaml and accept the documented "
            "leakage risk.")

    if split_level != "subject":
        # Image-level fallback: every file is its own "subject".
        subject_to_files = {f: [f] for f in files}
        unmatched = []

    units = sorted(subject_to_files.keys(),
                   key=lambda x: (int(x) if x.isdigit() else float("inf"), x))
    rng = random.Random(seed)
    rng.shuffle(units)

    n = len(units)
    n_test = max(1, round(n * test_fraction)) if n > 2 else max(0, n - 1)
    n_val = max(1, round(n * val_fraction)) if n > 2 else 0
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"Split leaves {n_train} training units out of {n}. "
            f"Lower val_fraction/test_fraction in config.yaml.")

    train_u = units[:n_train]
    val_u = units[n_train:n_train + n_val]
    test_u = units[n_train + n_val:]

    def files_for(unit_list):
        out = []
        for u in unit_list:
            out.extend(subject_to_files[u])
        return sorted(out)

    return {
        "seed": seed,
        "split_level": split_level,
        "n_total_subjects": n,
        "n_unmatched_filenames": len(unmatched),
        "unmatched_examples": unmatched[:10],
        "train_subjects": train_u,
        "val_subjects": val_u,
        "test_subjects": test_u,
        "train_files": files_for(train_u),
        "val_files": files_for(val_u),
        "test_files": files_for(test_u),
    }


def verify_no_leakage(split: dict) -> dict:
    """
    Assert the split is leak-free. Raises AssertionError with a specific
    message on any violation; returns a summary dict when clean.

    Three independent checks, because they can fail independently:
      1. no SUBJECT appears in more than one split
      2. no FILE appears in more than one split
      3. every file's parsed subject actually belongs to its split's subject set
         (catches a split file that was hand-edited or built by older code)
    """
    tr_s = set(split["train_subjects"])
    va_s = set(split["val_subjects"])
    te_s = set(split["test_subjects"])

    assert not (tr_s & va_s), f"Subject leakage train<->val: {sorted(tr_s & va_s)}"
    assert not (tr_s & te_s), f"Subject leakage train<->test: {sorted(tr_s & te_s)}"
    assert not (va_s & te_s), f"Subject leakage val<->test: {sorted(va_s & te_s)}"

    tr_f = set(split["train_files"])
    va_f = set(split["val_files"])
    te_f = set(split["test_files"])

    assert not (tr_f & va_f), f"File leakage train<->val: {sorted(tr_f & va_f)[:5]}"
    assert not (tr_f & te_f), f"File leakage train<->test: {sorted(tr_f & te_f)[:5]}"
    assert not (va_f & te_f), f"File leakage val<->test: {sorted(va_f & te_f)[:5]}"

    if split.get("split_level") == "subject":
        for name, files, subjects in (("train", tr_f, tr_s), ("val", va_f, va_s),
                                      ("test", te_f, te_s)):
            for f in files:
                subj = parse_subject(f)
                assert subj is None or subj in subjects, (
                    f"File '{f}' is in the {name} split but its subject '{subj}' "
                    f"is not in that split's subject list.")

    return {
        "leak_free": True,
        "split_level": split.get("split_level"),
        "subjects": {"train": len(tr_s), "val": len(va_s), "test": len(te_s)},
        "images": {"train": len(tr_f), "val": len(va_f), "test": len(te_f)},
    }
