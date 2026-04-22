"""Görsel I/O ve normalizasyon.

EXIF yönünü düzeltir, RGB/RGBA'ya çevirir, çözünürlüğü sınırlar ve dosya
doğrulaması yapar. Tüm pipeline numpy uint8 bekler.
"""

from __future__ import annotations

import io
from typing import Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.core.errors import InvalidImageError

MAX_SIDE = 1280
MIN_SIDE = 128

ImageSource = Union[str, bytes, io.BytesIO]


def _open(source: ImageSource) -> Image.Image:
    try:
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(source)
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"Görsel açılamadı: {exc}") from exc
    return ImageOps.exif_transpose(img)


def _resize_bounded(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _validate_size(img: Image.Image) -> None:
    w, h = img.size
    if min(w, h) < MIN_SIDE:
        raise InvalidImageError(
            f"Görsel çok küçük ({w}x{h}). En az {MIN_SIDE}px kısa kenar gerekir."
        )


def load_person_rgb(source: ImageSource) -> np.ndarray:
    """Manken fotoğrafını HxWx3 uint8 RGB olarak yükler."""
    img = _open(source).convert("RGB")
    img = _resize_bounded(img)
    _validate_size(img)
    return np.asarray(img, dtype=np.uint8)


def load_garment_rgba(source: ImageSource) -> np.ndarray:
    """Ürün görselini HxWx4 uint8 RGBA olarak yükler.

    Orijinal alfa varsa korunur; yoksa arka plan sonraki adımda kaldırılır.
    """
    img = _open(source)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img = _resize_bounded(img)
    _validate_size(img)
    return np.asarray(img, dtype=np.uint8)


def encode_png(rgb: np.ndarray) -> bytes:
    """Numpy RGB/RGBA'yı PNG byte'ına çevirir (indirme için)."""
    mode = "RGBA" if rgb.ndim == 3 and rgb.shape[2] == 4 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(rgb, mode=mode).save(buf, format="PNG")
    return buf.getvalue()
