"""İnsan maskesi çıkarımı.

Bu ortamda kurulu `mediapipe` paketi yalnızca `tasks` alt modülünü sağladığı
için `solutions` tabanlı segmentasyon kullanmıyoruz. Bunun yerine `rembg`
ile kişiyi arka plandan ayırıp compositing için bir silüet maskesi üretiyoruz.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.core.errors import PersonNotDetectedError

_PERSON_SESSION = None


def _get_person_session():
    global _PERSON_SESSION
    if _PERSON_SESSION is None:
        from rembg import new_session

        _PERSON_SESSION = new_session("u2netp")
    return _PERSON_SESSION


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """En büyük bağlı bileşeni korur; küçük artefaktları temizler."""
    try:
        import cv2

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        return np.where(labels == largest, 255, 0).astype(np.uint8)
    except Exception:
        return mask


def person_mask(person_rgb: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """HxW uint8 maske döndürür (255 = insan, 0 = arka plan)."""
    del threshold  # rembg alfa tabanlı çalışıyor; eşiği sabit kullanıyoruz.

    try:
        from rembg import remove

        session = _get_person_session()
        out = remove(Image.fromarray(person_rgb, mode="RGB"), session=session)
        if out.mode != "RGBA":
            out = out.convert("RGBA")
        alpha = np.asarray(out, dtype=np.uint8)[:, :, 3]
    except Exception as exc:
        raise PersonNotDetectedError(f"İnsan maskesi üretilemedi: {exc}") from exc

    mask = np.where(alpha > 16, 255, 0).astype(np.uint8)
    mask = _largest_component(mask)
    if np.count_nonzero(mask) < 500:
        raise PersonNotDetectedError(
            "Fotoğrafta yeterli insan silüeti çıkarılamadı. Daha net ve sade arka planlı bir fotoğraf deneyin."
        )

    return mask
