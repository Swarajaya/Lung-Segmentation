"""Dataset splitting and data-leakage prevention."""
import pytest

from src.splitting import build_split, verify_no_leakage, parse_subject, group_by_subject


def make_files(n_subjects=12, n_slices=5):
    return [f"train_Subject_{s}_{i}.png" for s in range(n_subjects) for i in range(n_slices)]


def test_parse_subject():
    assert parse_subject("train_Subject_17_287.png") == "17"
    assert parse_subject("val_Subject_3_9.png") == "3"
    assert parse_subject("random_file.png") is None


def test_group_by_subject_separates_unparsable():
    grouped, unmatched = group_by_subject(make_files(3, 2) + ["junk.png"])
    assert len(grouped) == 3
    assert unmatched == ["junk.png"]


def test_split_is_leak_free():
    split = build_split(make_files(), 0.15, 0.15, seed=42)
    summary = verify_no_leakage(split)
    assert summary["leak_free"] is True


def test_split_is_reproducible_from_seed():
    a = build_split(make_files(), 0.15, 0.15, seed=42)
    b = build_split(make_files(), 0.15, 0.15, seed=42)
    assert a["train_files"] == b["train_files"]
    assert a["test_subjects"] == b["test_subjects"]


def test_different_seed_gives_different_split():
    a = build_split(make_files(), 0.15, 0.15, seed=1)
    b = build_split(make_files(), 0.15, 0.15, seed=2)
    assert a["test_subjects"] != b["test_subjects"]


def test_every_file_appears_exactly_once():
    files = make_files()
    split = build_split(files, 0.15, 0.15, seed=42)
    combined = split["train_files"] + split["val_files"] + split["test_files"]
    assert sorted(combined) == sorted(files)
    assert len(combined) == len(set(combined))


def test_all_slices_of_a_subject_stay_together():
    split = build_split(make_files(), 0.15, 0.15, seed=42)
    for name in ("train", "val", "test"):
        subjects = set(split[f"{name}_subjects"])
        for f in split[f"{name}_files"]:
            assert parse_subject(f) in subjects


def test_verify_detects_injected_subject_leakage():
    split = build_split(make_files(), 0.15, 0.15, seed=42)
    split["train_subjects"] = split["train_subjects"] + [split["test_subjects"][0]]
    with pytest.raises(AssertionError, match="Subject leakage"):
        verify_no_leakage(split)


def test_verify_detects_injected_file_leakage():
    split = build_split(make_files(), 0.15, 0.15, seed=42)
    split["train_files"] = split["train_files"] + [split["test_files"][0]]
    with pytest.raises(AssertionError, match="File leakage"):
        verify_no_leakage(split)


def test_verify_detects_file_whose_subject_is_in_another_split():
    """A hand-edited split where a file's subject does not match its split."""
    split = build_split(make_files(), 0.15, 0.15, seed=42)
    stray = split["test_files"][0]
    split["test_files"] = split["test_files"][1:]
    split["train_files"] = split["train_files"] + [stray]
    with pytest.raises(AssertionError, match="is not in that split"):
        verify_no_leakage(split)


def test_subject_split_raises_when_no_subject_ids_parsable():
    with pytest.raises(ValueError, match="no subject IDs"):
        build_split(["a.png", "b.png", "c.png"], 0.15, 0.15, seed=42,
                    split_level="subject")


def test_synthetic_dataset_split_is_leak_free(synthetic_dataset):
    split = build_split(synthetic_dataset["images_dir"], 0.2, 0.2, seed=42)
    assert verify_no_leakage(split)["leak_free"] is True


def test_committed_split_is_leak_free(committed_split):
    """The split shipped in data/splits.json must itself be leak-free."""
    summary = verify_no_leakage(committed_split)
    assert summary["leak_free"] is True
    assert summary["split_level"] == "subject"
    assert summary["subjects"]["train"] > 0
    assert summary["subjects"]["val"] > 0
    assert summary["subjects"]["test"] > 0
