"""Torso (üst gövde) için inpaint maskesi üretimi.

Kişi silüetinden omuz ve kalça hizasını tahmin ederek ürünün
yerleşeceği bölgeyi boyar. Maske HxW uint8 (255 = inpaint edilecek,
0 = korunacak).
"""

from __future__ import annotations

import cv2
import numpy as np

from src.core.errors import PersonNotDetectedError
from src.services.person_parser import person_mask


def build_torso_mask(
    person_rgb: np.ndarray,
    top_ratio: float = 0.18,
    bottom_ratio: float = 0.62,
    side_inset: float = 0.04,
    feather: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Kişi + torso maskesi döndürür.

    - `person_full`: Kişi silüeti (HxW uint8, 255 insan)
    - `torso_mask`: Inpaint edilecek üst gövde bölgesi (HxW uint8)
    """
    silhouette = person_mask(person_rgb)

    ys, xs = np.where(silhouette > 0)
    if ys.size == 0:
        raise PersonNotDetectedError("Kişi silüeti bulunamadı.")
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    box_h = max(y1 - y0, 1)
    box_w = max(x1 - x0, 1)

    top = y0 + int(box_h * top_ratio)
    bottom = y0 + int(box_h * bottom_ratio)
    inset = int(box_w * side_inset)

    torso = np.zeros_like(silhouette)
    torso[top:bottom, x0 + inset : x1 - inset] = 255
    # Silüetin dışına taşmasın
    torso = cv2.bitwise_and(torso, silhouette)

    if feather > 0:
        torso = cv2.GaussianBlur(torso, (feather | 1, feather | 1), 0)
        # Tam beyaz kısımları koru; kenarları yumuşak bırak
        torso = np.clip(torso, 0, 255).astype(np.uint8)

    if int(np.count_nonzero(torso > 32)) < 400:
        raise PersonNotDetectedError(
            "Torso maskesi yeterince büyük çıkmadı. Daha yakın ve ön cepheli bir manken görseli deneyin."
        )

    return silhouette, torso
