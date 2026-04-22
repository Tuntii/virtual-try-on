"""Girdi kalite kontrolleri.

Model kalitesi kadar input kalitesi de sonucu belirliyor. Bu modül,
ürün ve manken görsellerini hızlıca inceleyip kullanıcıya anlamlı
uyarılar üretir.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.services.garment_mask import crop_to_alpha, ensure_garment_rgba


def check_garment(garment_rgba: np.ndarray) -> list[str]:
    """Ürün görseli packshot kalitesinde mi? Uyarı listesi döndürür."""
    warnings: list[str] = []
    try:
        rgba = ensure_garment_rgba(garment_rgba)
        rgba = crop_to_alpha(rgba)
    except Exception as exc:
        return [f"Ürün görseli işlenemedi: {exc}"]

    alpha = rgba[:, :, 3]
    h, w = alpha.shape[:2]

    # 1) Doluluk oranı çok düşükse muhtemelen ürün çok küçük veya hatalı maske
    fill = float(np.count_nonzero(alpha > 32)) / float(h * w + 1)
    if fill < 0.15:
        warnings.append(
            "Ürün kadrajın çok küçük bir kısmını kaplıyor. Daha sıkı kırpılmış bir packshot tercih edin."
        )

    # 2) En-boy oranı uçta mı?
    ratio = h / float(w) if w else 0.0
    if ratio > 2.2 or ratio < 0.45:
        warnings.append(
            "Ürün görseli çok dar/uzun. Kare-ya-kın veya dikey packshot idealdir."
        )

    # 3) Birden fazla büyük nesne var mı? (ör. el, kol, insan da çekilmiş)
    mask = (alpha > 32).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size >= 2:
            areas_sorted = np.sort(areas)[::-1]
            if areas_sorted[1] > max(500, 0.08 * areas_sorted[0]):
                warnings.append(
                    "Ürün görselinde birden fazla nesne algılandı (örn. el, kol veya manken). "
                    "Yalnızca ürünün olduğu, düz zeminli bir fotoğraf kalite için çok daha iyidir."
                )

    # 4) Çok yüksek çözünürlük yoksa uyar
    if min(h, w) < 300:
        warnings.append("Ürün görseli çözünürlüğü düşük. Daha yüksek çözünürlük daha iyi sonuç verir.")

    return warnings


def check_person(person_rgb: np.ndarray) -> list[str]:
    """Manken fotoğrafı ön cepheli ve yeterince büyük mü?"""
    warnings: list[str] = []
    h, w = person_rgb.shape[:2]
    if min(h, w) < 384:
        warnings.append("Manken fotoğrafı küçük. En az 512px kısa kenar önerilir.")

    # Görüntü çok geniş panorama ise uygun değildir.
    if w / float(h) > 1.6:
        warnings.append("Fotoğraf yatay panoramik görünüyor; dikey/dikdörtgen portre daha iyidir.")

    return warnings
