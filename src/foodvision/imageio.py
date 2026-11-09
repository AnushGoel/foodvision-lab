"""Validation of user-supplied image bytes (PIL only, no PyTorch needed)."""

from __future__ import annotations

import io

from PIL import Image, ImageOps

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP"})
MAX_BYTES = 15 * 1024 * 1024
MIN_SIDE = 32
MAX_PIXELS = 40_000_000


class InvalidImageError(ValueError):
    """Raised with a user-facing message when an upload cannot be used."""


def load_image(data: bytes) -> Image.Image:
    """Decode and validate uploaded bytes; returns an RGB image with EXIF orientation applied."""
    if not data:
        raise InvalidImageError("The file is empty. Choose a JPEG, PNG, WebP or BMP photo.")
    if len(data) > MAX_BYTES:
        raise InvalidImageError(f"The file is larger than {MAX_BYTES // 2**20} MB. Upload a smaller photo.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            width, height = probe.size
            probe.verify()
        if fmt not in ALLOWED_FORMATS:
            raise InvalidImageError(f"{fmt} files are not supported. Upload a JPEG, PNG, WebP or BMP photo.")
        if width * height > MAX_PIXELS:
            raise InvalidImageError("The image has too many pixels. Upload a photo under 40 megapixels.")
        decoded = Image.open(io.BytesIO(data))
        decoded.load()
    except InvalidImageError:
        raise
    except Exception as error:  # truncated, corrupted or non-image data
        raise InvalidImageError("This file could not be read as an image. It may be corrupted.") from error
    if min(decoded.size) < MIN_SIDE:
        raise InvalidImageError(f"The image is smaller than {MIN_SIDE} pixels on one side.")
    oriented: Image.Image = ImageOps.exif_transpose(decoded)  # honour the camera's rotation flag
    if oriented.mode in ("RGBA", "LA") or (oriented.mode == "P" and "transparency" in oriented.info):
        rgba = oriented.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return oriented.convert("RGB")
