import shutil

from PIL import Image

from foodvision.audit import audit_dataset, dhash, hamming


def test_dhash_is_resize_robust():
    image = Image.effect_mandelbrot((256, 256), (-2, -1.5, 1, 1.5), 50).convert("RGB")
    assert hamming(dhash(image), dhash(image.resize((128, 128)))) <= 6


def test_clean_folder_summary(image_root):
    summary, frame = audit_dataset(image_root)
    assert summary["n_images"] == 27
    assert summary["counts"]["train"] == {"pizza": 6, "steak": 6, "sushi": 6}
    assert summary["n_corrupted"] == 0
    assert summary["cross_split_pairs"] == 0
    assert set(frame.columns) >= {"width", "height", "aspect_ratio", "md5", "dhash"}


def test_detects_corruption_and_cross_split_leakage(image_root):
    leaked = image_root / "test" / "pizza" / "leaked_copy.jpg"
    shutil.copy(image_root / "train" / "pizza" / "pizza_0.jpg", leaked)
    (image_root / "train" / "steak" / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    summary, _ = audit_dataset(image_root)
    assert summary["n_corrupted"] == 1
    assert summary["exact_duplicate_pairs"] >= 1
    assert summary["cross_split_pairs"] >= 1
