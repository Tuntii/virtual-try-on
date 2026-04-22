"""Silüet tabanlı gövde landmark tahmini.

Bu ortamda `mediapipe.solutions` bulunmadığı için landmark'ları kişi
maskesinden yaklaşık olarak tahmin ediyoruz. POC için yeterince iyi bir
omuz/kalça hizalama sağlar.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.core.errors import PersonNotDetectedError
from src.services.person_parser import person_mask


@dataclass
class PoseLandmarks:
    left_shoulder: tuple[float, float]
    right_shoulder: tuple[float, float]
    left_hip: tuple[float, float]
    right_hip: tuple[float, float]
    image_size: tuple[int, int]  # (w, h)

    @property
    def shoulder_width_px(self) -> float:
        lx, ly = self.left_shoulder
        rx, ry = self.right_shoulder
        return float(np.hypot(lx - rx, ly - ry))

    @property
    def torso_center(self) -> tuple[float, float]:
        xs = [self.left_shoulder[0], self.right_shoulder[0], self.left_hip[0], self.right_hip[0]]
        ys = [self.left_shoulder[1], self.right_shoulder[1], self.left_hip[1], self.right_hip[1]]
        return float(np.mean(xs)), float(np.mean(ys))

    @property
    def shoulder_midpoint(self) -> tuple[float, float]:
        return (
            float((self.left_shoulder[0] + self.right_shoulder[0]) / 2.0),
            float((self.left_shoulder[1] + self.right_shoulder[1]) / 2.0),
        )

    @property
    def hip_midpoint(self) -> tuple[float, float]:
        return (
            float((self.left_hip[0] + self.right_hip[0]) / 2.0),
            float((self.left_hip[1] + self.right_hip[1]) / 2.0),
        )


def _row_edges(mask: np.ndarray, row_index: int) -> tuple[float, float] | None:
    row = mask[row_index]
    xs = np.where(row > 0)[0]
    if xs.size == 0:
        return None
    return float(xs.min()), float(xs.max())


def _nearest_edges(mask: np.ndarray, target_row: int, max_radius: int = 25) -> tuple[int, tuple[float, float]] | None:
    h = mask.shape[0]
    for radius in range(max_radius + 1):
        for row in (target_row - radius, target_row + radius):
            if row < 0 or row >= h:
                continue
            edges = _row_edges(mask, row)
            if edges is not None:
                return row, edges
    return None


def _estimate_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        raise PersonNotDetectedError(
            "Fotoğrafta insan silüeti bulunamadı. Daha net bir ön cepheli görsel deneyin."
        )
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def estimate_pose(person_rgb: np.ndarray) -> PoseLandmarks:
    h, w = person_rgb.shape[:2]
    mask = person_mask(person_rgb)
    x0, y0, x1, y1 = _estimate_bbox(mask)

    box_h = max(y1 - y0, 1)
    box_w = max(x1 - x0, 1)
    if box_h < 80 or box_w < 40:
        raise PersonNotDetectedError(
            "Kişi silüeti çok küçük görünüyor. Daha büyük ve tam gövdeli bir fotoğraf deneyin."
        )

    shoulder_target = y0 + int(box_h * 0.22)
    hip_target = y0 + int(box_h * 0.58)

    shoulder_hit = _nearest_edges(mask, shoulder_target, max_radius=max(20, box_h // 8))
    hip_hit = _nearest_edges(mask, hip_target, max_radius=max(20, box_h // 8))
    if shoulder_hit is None or hip_hit is None:
        raise PersonNotDetectedError(
            "Omuz/kalça konumu tahmin edilemedi. Tam gövde ve sade arka planlı bir fotoğraf deneyin."
        )

    sy, (sx0, sx1) = shoulder_hit
    hy, (hx0, hx1) = hip_hit

    shoulder_width = sx1 - sx0
    hip_width = hx1 - hx0
    if shoulder_width < 20 or hip_width < 20:
        raise PersonNotDetectedError(
            "Silüet yeterince net değil; omuz/kalça genişliği çok dar kaldı."
        )

    # Hafif içeri alarak kolların etkisini azalt.
    shoulder_inset = shoulder_width * 0.08
    hip_inset = hip_width * 0.12

    ls = (sx0 + shoulder_inset, float(sy))
    rs = (sx1 - shoulder_inset, float(sy))
    lh = (hx0 + hip_inset, float(hy))
    rh = (hx1 - hip_inset, float(hy))

    return PoseLandmarks(
        left_shoulder=ls,
        right_shoulder=rs,
        left_hip=lh,
        right_hip=rh,
        image_size=(w, h),
    )
