"""OpenAI gpt-image-1 tabanlı virtual try-on backend.

Kişi fotoğrafı + ürün görseli → gpt-image-1 images.edit API → giydirilmiş sonuç.

Kullanım:
    OPENAI_API_KEY ortam değişkeni set edilmeli, ya da app'te girilmeli.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.backends.base import TryOnBackend
from src.core.errors import TryOnError
from src.core.types import TryOnRequest, TryOnResult
from src.services.garment_mask import crop_to_alpha, ensure_garment_rgba
from src.services.torso_mask import build_torso_mask


def _resize_max(arr: np.ndarray, max_px: int) -> np.ndarray:
    """Görüntüyü max_px piksel uzun kenar olacak şekilde küçültür."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_px:
        return arr
    scale = max_px / max(h, w)
    nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
    return cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)


def _resize_max_gray(mask: np.ndarray, max_px: int, orig_shape: tuple) -> np.ndarray:
    """Gri maske'yi kişi görseli ile aynı orana küçültür."""
    orig_h, orig_w = orig_shape
    if max(orig_h, orig_w) <= max_px:
        return mask
    scale = max_px / max(orig_h, orig_w)
    nw, nh = max(int(orig_w * scale), 1), max(int(orig_h * scale), 1)
    return cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_LINEAR)


def _to_png_bytes(arr: np.ndarray) -> bytes:
    """NumPy array → PNG bayt dizisi."""
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _build_mask_png(torso_mask: np.ndarray) -> bytes:
    """Torso maskesinden OpenAI mask PNG'si üretir.

    OpenAI kuralı: şeffaf (alpha=0) alan → düzenlenecek bölge.
    torso_mask: grayscale, 255 = torso (düzenlenecek), 0 = koru.
    """
    h, w = torso_mask.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Torso olmayan alanlar opak (korunacak), torso alanı şeffaf (düzenlenecek)
    keep = torso_mask < 128
    rgba[keep, 3] = 255   # opaque = koru
    rgba[~keep, 3] = 0    # transparent = düzenle
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _square_pad_png(png_bytes: bytes) -> bytes:
    """Görüntüyü kare hale getirir (OpenAI mask kare zorunluluğu için)."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    w, h = img.size
    if w == h:
        return png_bytes
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def _square_pad_rgb_png(png_bytes: bytes) -> bytes:
    """RGB görüntüyü beyaz dolgu ile kare hale getirir."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    w, h = img.size
    if w == h:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


class OpenAiImageBackend(TryOnBackend):
    """gpt-image-1 images.edit ile virtual try-on."""

    name = "openai-edit"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-image-2",
        size: str = "auto",
        quality: str = "medium",
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.size = size
        self.quality = quality
        self._client: Any | None = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import httpx  # type: ignore
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise TryOnError(
                "openai paketi kurulu değil. `pip install openai>=1.0 httpx` ile kurun."
            ) from exc

        if not self._api_key:
            raise TryOnError(
                "OpenAI API anahtarı bulunamadı. "
                "OPENAI_API_KEY ortam değişkenini set edin veya uygulamada girin."
            )
        # Uzun süreli yükleme için geniş timeout; büyük görüntü dosyalarında
        # default 60s timeout 'Connection error' olarak yansır.
        self._client = OpenAI(
            api_key=self._api_key,
            http_client=httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0)),
        )

    # ------------------------------------------------------------------
    def run(self, request: TryOnRequest) -> TryOnResult:
        self._ensure_client()

        person_rgb = request.person_rgb
        options = request.options or {}
        quality = options.get("quality", self.quality)
        size = options.get("size", self.size)
        prompt = options.get("prompt", self._default_prompt(request.garment_name))

        # 1) Torso maskesi
        _silhouette, torso = build_torso_mask(person_rgb)

        # 2) Kişi PNG — max 1024px (API limiti ve yükleme hızı için)
        person_png = _square_pad_rgb_png(_to_png_bytes(_resize_max(person_rgb, 1024)))

        # 3) Mask PNG (şeffaf = düzenle = torso alanı)
        torso_resized = _resize_max_gray(torso, 1024, person_rgb.shape[:2])
        mask_png = _square_pad_png(_build_mask_png(torso_resized))

        # 4) Ürün görseli PNG (beyaz zemin, max 1024px)
        garment = ensure_garment_rgba(request.garment_rgba)
        garment = crop_to_alpha(garment)
        g_rgb = garment[:, :, :3].astype(np.float32)
        g_alpha = garment[:, :, 3:4].astype(np.float32) / 255.0
        garment_rgb = (g_rgb * g_alpha + 255.0 * (1.0 - g_alpha)).clip(0, 255).astype(np.uint8)
        garment_png = _square_pad_rgb_png(_to_png_bytes(_resize_max(garment_rgb, 1024)))

        # 5) API çağrısı
        # image parametresi: BytesIO listesi — nested tuple formatı SDK'da çalışmıyor.
        try:
            response = self._client.images.edit(
                model=self.model,
                image=[
                    ("person.png", io.BytesIO(person_png), "image/png"),
                    ("garment.png", io.BytesIO(garment_png), "image/png"),
                ],
                mask=("mask.png", io.BytesIO(mask_png), "image/png"),
                prompt=prompt,
                size=size,
                quality=quality,
            )
        except Exception as exc:
            # Daha bilgilendirici hata mesajı
            msg = str(exc)
            if "Connection" in msg or "connect" in msg.lower():
                raise TryOnError(
                    f"OpenAI bağlantı hatası: {exc}\n"
                    "API sunucusuna ulaşılamıyor. VPN veya proxy gerekli olabilir."
                ) from exc
            if "401" in msg or "authentication" in msg.lower():
                raise TryOnError(f"OpenAI API anahtarı geçersiz: {exc}") from exc
            raise TryOnError(f"OpenAI API hatası: {exc}") from exc

        # 6) Sonucu decode et
        try:
            img_data = response.data[0]
            if getattr(img_data, "b64_json", None):
                raw = base64.b64decode(img_data.b64_json)
            elif getattr(img_data, "url", None):
                import urllib.request
                with urllib.request.urlopen(img_data.url) as resp:  # noqa: S310
                    raw = resp.read()
            else:
                raise TryOnError("API yanıtı görüntü içermiyor.")
            result_img = Image.open(io.BytesIO(raw)).convert("RGB")
            # Orijinal en-boy oranına geri döndür
            orig_h, orig_w = person_rgb.shape[:2]
            result_img = result_img.resize((orig_w, orig_h), Image.LANCZOS)
            result_arr = np.asarray(result_img, dtype=np.uint8)
        except TryOnError:
            raise
        except Exception as exc:
            raise TryOnError(f"Sonuç decode hatası: {exc}") from exc

        return TryOnResult(
            composite_rgb=result_arr,
            person_name=request.person_name,
            garment_name=request.garment_name,
            debug={
                "backend": self.name,
                "model": self.model,
                "quality": quality,
                "size": size,
            },
        )

    @staticmethod
    def _default_prompt(garment_name: str) -> str:
        name_hint = garment_name.replace("-", " ").replace("_", " ") if garment_name else "giysi"
        return (
            f"The person in the first image is wearing the '{name_hint}' garment shown in the "
            "second image. Preserve the person's face, hair, skin tone, pose, and the background "
            "exactly as they are. Only replace the clothing on the torso/upper body area. "
            "The result should look like a real fashion photograph with natural lighting and "
            "realistic fabric draping."
        )
