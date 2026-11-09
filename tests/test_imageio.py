import io

import pytest
from PIL import Image

from foodvision.imageio import InvalidImageError, load_image


def _encode(image, fmt):
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


def test_accepts_jpeg_and_returns_rgb():
    image = load_image(_encode(Image.new("RGB", (320, 240), "red"), "JPEG"))
    assert image.mode == "RGB" and image.size == (320, 240)


def test_transparent_png_is_composited_on_white():
    image = load_image(_encode(Image.new("RGBA", (64, 64), (0, 0, 0, 0)), "PNG"))
    assert image.getpixel((1, 1)) == (255, 255, 255)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"definitely not an image",
        _encode(Image.new("RGB", (320, 240), "red"), "JPEG")[:300],
        _encode(Image.new("RGB", (10, 10), "red"), "PNG"),
        _encode(Image.new("RGB", (64, 64), "red"), "GIF"),
    ],
    ids=["empty", "text", "truncated", "too-small", "unsupported-gif"],
)
def test_rejects_unusable_uploads_with_a_message(data):
    with pytest.raises(InvalidImageError) as error:
        load_image(data)
    assert str(error.value)
