"""CPU tabanlı compositing backend'i.

Pose landmark'larına göre ürünü ölçekler, omuz hattına göre döndürür ve
insan maskesiyle birlikte alfa karıştırma yaparak sonuç üretir. POC için
doğal görünüm sağlar; gerçek generative VTON değildir.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.backends.base import TryOnBackend
from src.core.errors import GarmentMaskError, PersonNotDetectedError, TryOnError
from src.core.types import TryOnRequest, TryOnResult
from src.services.garment_mask import crop_to_alpha, ensure_garment_rgba
from src.services.person_parser import person_mask
from src.services.pose_estimator import PoseLandmarks, estimate_pose


# Omuz genişliğine göre ürünün hedef genişliği çarpanı.
# Stüdyo çekimi düz ürünlerde ~1.9-2.2 arası doğal sonuç verir.
DEFAULT_WIDTH_SCALE = 2.05
# Yaka çizgisini omuz hattının biraz üzerine yerleştirmek için dikey kayma
# (omuz genişliğine oranla).
DEFAULT_NECK_OFFSET = 0.08


def _rotate_rgba(rgba: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy
    return cv2.warpAffine(
        rgba,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _alpha_composite(
    base_rgb: np.ndarray,
    overlay_rgba: np.ndarray,
    top_left: tuple[int, int],
    clip_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`overlay_rgba`'yı `base_rgb` üzerine yerleştirir; `clip_mask` (HxW uint8)
    verilmişse alfa bu maskeyle çarpılır (insan silüetinin dışına taşma önlenir).
    """
    H, W = base_rgb.shape[:2]
    oh, ow = overlay_rgba.shape[:2]
    x0, y0 = top_left
    x1, y1 = x0 + ow, y0 + oh

    # Base içindeki kesim bölgesi
    bx0, by0 = max(x0, 0), max(y0, 0)
    bx1, by1 = min(x1, W), min(y1, H)
    if bx0 >= bx1 or by0 >= by1:
        return base_rgb.copy(), np.zeros((H, W), dtype=np.uint8)

    # Overlay içindeki kesim bölgesi
    ox0, oy0 = bx0 - x0, by0 - y0
    ox1, oy1 = ox0 + (bx1 - bx0), oy0 + (by1 - by0)

    overlay_crop = overlay_rgba[oy0:oy1, ox0:ox1]
    alpha = overlay_crop[:, :, 3].astype(np.float32) / 255.0

    if clip_mask is not None:
        clip = clip_mask[by0:by1, bx0:bx1].astype(np.float32) / 255.0
        alpha = alpha * clip

    alpha3 = alpha[:, :, None]
    base_crop = base_rgb[by0:by1, bx0:bx1].astype(np.float32)
    overlay_rgb = overlay_crop[:, :, :3].astype(np.float32)

    blended = overlay_rgb * alpha3 + base_crop * (1.0 - alpha3)
    out = base_rgb.copy()
    out[by0:by1, bx0:bx1] = np.clip(blended, 0, 255).astype(np.uint8)

    placement = np.zeros((H, W), dtype=np.uint8)
    placement[by0:by1, bx0:bx1] = (alpha * 255).astype(np.uint8)
    return out, placement


def _shoulder_angle_deg(pose: PoseLandmarks) -> float:
    # Ürün PNG'leri dik duruşlu çekildiği için doğrudan omuz hattı açısı
    # yeterli bir yaklaşıktır.
    lx, ly = pose.left_shoulder
    rx, ry = pose.right_shoulder
    # left_shoulder ekran-sol tarafta görünür; mediapipe "LEFT" kullanıcı sol
    # olduğu için ekranda sağdadır. Yine de simetrik açı alıyoruz.
    dx = lx - rx
    dy = ly - ry
    return math.degrees(math.atan2(dy, dx))


class CpuCompositeBackend(TryOnBackend):
    name = "cpu-composite"

    def run(self, request: TryOnRequest) -> TryOnResult:
        warnings: list[str] = []
        options = request.options or {}
        width_scale = float(options.get("width_scale", DEFAULT_WIDTH_SCALE))
        neck_offset = float(options.get("neck_offset", DEFAULT_NECK_OFFSET))
        use_person_mask = bool(options.get("use_person_mask", True))
        apply_rotation = bool(options.get("apply_rotation", True))

        person_rgb = request.person_rgb
        garment_rgba = request.garment_rgba

        # 1) Ürün maskesini garanti et ve sıkı kırp
        try:
            garment_rgba = ensure_garment_rgba(garment_rgba)
            garment_rgba = crop_to_alpha(garment_rgba)
        except GarmentMaskError:
            raise
        except Exception as exc:  # pragma: no cover
            raise TryOnError(f"Ürün işleme hatası: {exc}") from exc

        # 2) Pose
        try:
            pose = estimate_pose(person_rgb)
        except PersonNotDetectedError:
            raise

        shoulder_w = pose.shoulder_width_px
        if shoulder_w < 20:
            raise PersonNotDetectedError(
                "Omuzlar yeterince geniş görünmüyor. Daha yakın/ön cepheli bir fotoğraf deneyin."
            )

        # 3) Ürünü hedef genişliğe ölçekle
        target_w = max(int(shoulder_w * width_scale), 32)
        gh, gw = garment_rgba.shape[:2]
        scale = target_w / float(gw)
        target_h = max(int(gh * scale), 32)
        garment_resized = cv2.resize(
            garment_rgba, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

        # 4) Döndür
        if apply_rotation:
            angle = _shoulder_angle_deg(pose)
            # Çok ufak açıları bırakmak sonucu doğallaştırır
            if abs(angle) < 1.0:
                angle = 0.0
            garment_oriented = _rotate_rgba(garment_resized, -angle)
        else:
            garment_oriented = garment_resized

        # 5) Yerleştirme noktası: ürünün üst-orta noktası omuz orta noktasının
        # hemen üstüne denk gelsin (yaka boşluğu için küçük bir offset).
        smx, smy = pose.shoulder_midpoint
        oy = int(smy - neck_offset * shoulder_w)
        oh2, ow2 = garment_oriented.shape[:2]
        top_left = (int(smx - ow2 / 2.0), int(oy - oh2 * 0.08))

        # 6) İsteğe bağlı insan silüeti maskesi
        clip = None
        if use_person_mask:
            try:
                clip = person_mask(person_rgb)
                # maskeyi biraz yumuşat (sert kenarlara karşı)
                clip = cv2.GaussianBlur(clip, (9, 9), 0)
            except Exception:  # pragma: no cover
                warnings.append("İnsan maskesi üretilemedi, yerleştirme maskesiz yapıldı.")

        composite, placement = _alpha_composite(
            person_rgb, garment_oriented, top_left, clip_mask=clip
        )

        debug = {
            "shoulder_width_px": round(shoulder_w, 1),
            "target_garment_width": target_w,
            "angle_deg": round(_shoulder_angle_deg(pose), 2),
            "top_left": top_left,
            "width_scale": width_scale,
            "neck_offset": neck_offset,
        }

        return TryOnResult(
            composite_rgb=composite,
            debug=debug,
            warnings=warnings,
            person_name=request.person_name,
            garment_name=request.garment_name,
            preview_mask=placement,
        )
