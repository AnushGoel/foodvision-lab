from collections import Counter

from foodvision.data import class_names, index_split, stratified_split


def test_index_and_class_mapping(image_root):
    classes = class_names(image_root / "train")
    assert classes == ["pizza", "steak", "sushi"]
    samples = index_split(image_root / "train", classes)
    assert len(samples) == 18
    assert {s.label for s in samples} == {0, 1, 2}


def test_stratified_split_is_disjoint_complete_balanced_and_reproducible(image_root):
    samples = index_split(image_root / "train", class_names(image_root / "train"))
    train, val = stratified_split(samples, 1 / 3, seed=42)
    assert not {s.path for s in train} & {s.path for s in val}
    assert len(train) + len(val) == len(samples)
    assert Counter(s.label for s in val) == {0: 2, 1: 2, 2: 2}
    assert stratified_split(samples, 1 / 3, seed=42) == (train, val)
    assert stratified_split(samples, 1 / 3, seed=1) != (train, val)
