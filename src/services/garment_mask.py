"""Ürün görseli maskesi.

Alfa kanalı zaten mevcutsa onu kullanır; değilse `rembg` ile arka planı kaldırır.
Sonuç RGBA uint8 numpy dizisidir.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.core.errors import GarmentMaskError

_REMBG_SESSION = None


def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        try:
            from rembg import new_session

            _REMBG_SESSION = new_session("u2netp")  # küçük ve CPU dostu
        except Exception as exc:  # pragma: no cover
            raise GarmentMaskError(f"rembg başlatılamadı: {exc}") from exc
    return _REMBG_SESSION


def _has_useful_alpha(rgba: np.ndarray) -> bool:
    if rgba.shape[2] != 4:
        return False
    alpha = rgba[:, :, 3]
    # Alfa gerçekten kullanılıyor mu? (tamamen 255 değilse evet)
    return bool(np.any(alpha < 250))


def ensure_garment_rgba(garment_rgba: np.ndarray) -> np.ndarray:
    """RGBA garanti eder. Gerekirse arka planı rembg ile kaldırır."""
    if _has_useful_alpha(garment_rgba):
        return garment_rgba

    try:
        from rembg import remove

        session = _get_rembg_session()
        pil = Image.fromarray(garment_rgba, mode="RGBA")
        out = remove(pil, session=session)
        if out.mode != "RGBA":
            out = out.convert("RGBA")
        arr = np.asarray(out, dtype=np.uint8)
    except Exception as exc:
        raise GarmentMaskError(f"Ürün arka planı kaldırılamadı: {exc}") from exc

    if not _has_useful_alpha(arr):
        raise GarmentMaskError(
            "Ürün maskesi üretilemedi. Düz zemin veya şeffaf PNG ile tekrar deneyin."
        )
    return arr


def crop_to_alpha(rgba: np.ndarray, pad: int = 4) -> np.ndarray:
    """Alfa olan bölgeye göre sıkı kırpar."""
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if ys.size == 0 or xs.size == 0:
        raise GarmentMaskError("Ürün maskesi boş.")
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, rgba.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, rgba.shape[1])
    return rgba[y0:y1, x0:x1]
