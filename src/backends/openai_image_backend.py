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


def _dominant_kmeans(rgb_arr: np.ndarray, k: int = 3) -> np.ndarray:
    """K-means ile baskın renk merkezlerini döndürür (kx3 float32)."""
    small = cv2.resize(rgb_arr, (64, 64)).reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        small, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(counts)[::-1]
    return centers[order]  # boyuta göre sıralı


def _dominant_color_desc(rgb_arr: np.ndarray, top_n: int = 2) -> str:
    """Garment'ın baskın renklerini hex string olarak döndürür (prompt'a eklemek için)."""
    centers = _dominant_kmeans(rgb_arr, k=top_n)
    parts = []
    for c in centers:
        r, g, b = int(c[0]), int(c[1]), int(c[2])
        parts.append(f"#{r:02x}{g:02x}{b:02x}")
    return ", ".join(parts)


def _score_result(
    result_rgb: np.ndarray,
    person_rgb: np.ndarray,
    torso_mask: np.ndarray,
    garment_centers: np.ndarray,
) -> float:
    """0-1 arası kalite skoru hesaplar.

    İki bileşen:
    - ``bg_score``: Torso dışı arka planın kişi orijinaliyle benzerliği (yüz,
      saç, zemin korunmalı).
    - ``color_score``: Torso bölgesinin garment baskın renklerine yakınlığı.
    """
    orig_h, orig_w = person_rgb.shape[:2]

    # Sonucu orijinal boyuta getir
    if result_rgb.shape[:2] != (orig_h, orig_w):
        result_rgb = cv2.resize(result_rgb, (orig_w, orig_h))

    # Torso maskesini boolean'a çevir
    if torso_mask.shape[:2] != (orig_h, orig_w):
        torso_resized = cv2.resize(torso_mask, (orig_w, orig_h))
    else:
        torso_resized = torso_mask
    torso_bin = torso_resized > 32

    # Arka plan koruma skoru (torso dışı alanlar)
    bg_mask = ~torso_bin
    if bg_mask.sum() > 200:
        diff = np.abs(
            result_rgb[bg_mask].astype(np.float32)
            - person_rgb[bg_mask].astype(np.float32)
        )
        bg_score = 1.0 - float(diff.mean()) / 255.0
    else:
        bg_score = 0.8  # belirsizse orta

    # Garment renk varlığı skoru (torso bölgesi)
    if torso_bin.sum() > 200 and garment_centers is not None and len(garment_centers) > 0:
        torso_pixels = result_rgb[torso_bin].astype(np.float32)  # Nx3
        min_dists = np.stack(
            [np.linalg.norm(torso_pixels - c[None, :], axis=1) for c in garment_centers],
            axis=0,
        ).min(axis=0)  # N
        # Garment rengine 70 birim yakın piksellerin oranı
        color_score = float((min_dists < 70).mean())
    else:
        color_score = 0.5

    # Arka plan korunması daha kritik (60/40)
    return 0.60 * bg_score + 0.40 * color_score


class OpenAiImageBackend(TryOnBackend):
    """gpt-image-1 images.edit ile virtual try-on."""

    name = "openai-edit"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-image-1",
        size: str = "auto",
        quality: str = "medium",
        num_samples: int = 2,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.size = size
        self.quality = quality
        self.num_samples = max(1, int(num_samples))
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

        # Key yoksa tekrar dene: önce env var, sonra .streamlit/secrets.toml
        if not self._api_key:
            self._api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            try:
                import tomllib  # type: ignore
                toml_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", ".streamlit", "secrets.toml"
                )
                with open(os.path.normpath(toml_path), "rb") as f:
                    self._api_key = tomllib.load(f).get("OPENAI_API_KEY", "")
            except Exception:
                pass
        if not self._api_key:
            raise TryOnError(
                "OpenAI API anahtarı bulunamadı. "
                ".streamlit/secrets.toml veya OPENAI_API_KEY env değişkenini kontrol edin."
            )
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
        num_samples = int(options.get("num_samples", self.num_samples))

        # 1) Torso maskesi
        _silhouette, torso = build_torso_mask(person_rgb)

        # 2) Kişi PNG — max 1024px
        person_png = _square_pad_rgb_png(_to_png_bytes(_resize_max(person_rgb, 1024)))

        # 3) Mask PNG (şeffaf = düzenle = torso alanı)
        torso_resized = _resize_max_gray(torso, 1024, person_rgb.shape[:2])
        mask_png = _square_pad_png(_build_mask_png(torso_resized))

        # 4) Ürün görseli PNG + renk analizi
        garment = ensure_garment_rgba(request.garment_rgba)
        garment = crop_to_alpha(garment)
        g_rgb = garment[:, :, :3].astype(np.float32)
        g_alpha = garment[:, :, 3:4].astype(np.float32) / 255.0
        garment_rgb = (g_rgb * g_alpha + 255.0 * (1.0 - g_alpha)).clip(0, 255).astype(np.uint8)
        garment_png = _square_pad_rgb_png(_to_png_bytes(_resize_max(garment_rgb, 1024)))

        # Garment baskın renkleri (prompt + skorlama için)
        garment_centers = _dominant_kmeans(garment_rgb, k=3)
        color_hint = _dominant_color_desc(garment_rgb, top_n=2)

        prompt = options.get(
            "prompt", self._default_prompt(request.garment_name, color_hint)
        )

        # 5) Best-of-N örnekleme: her API çağrısı sequential (model n=1 desteklediğinde
        #    num_samples doğrudan n= parametresine geçirilebilir)
        candidates: list[np.ndarray] = []
        scores: list[float] = []
        errors: list[str] = []

        for attempt in range(num_samples):
            try:
                response = self._client.images.edit(
                    model=self.model,
                    image=[
                        ("person.png", io.BytesIO(person_png), "image/png"),
                        ("garment.png", io.BytesIO(garment_png), "image/png"),
                    ],
                    mask=io.BytesIO(mask_png),
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    response_format="b64_json",
                )
                raw = self._decode_response(response)
                result_img = Image.open(io.BytesIO(raw)).convert("RGB")
                orig_h, orig_w = person_rgb.shape[:2]
                result_img = result_img.resize((orig_w, orig_h), Image.LANCZOS)
                result_arr = np.asarray(result_img, dtype=np.uint8)

                score = _score_result(result_arr, person_rgb, torso, garment_centers)
                candidates.append(result_arr)
                scores.append(score)
            except TryOnError:
                raise
            except Exception as exc:
                msg = str(exc)
                if "Connection" in msg or "connect" in msg.lower():
                    raise TryOnError(
                        f"OpenAI bağlantı hatası: {exc}\n"
                        "API sunucusuna ulaşılamıyor. VPN veya proxy gerekli olabilir."
                    ) from exc
                if "401" in msg or "authentication" in msg.lower():
                    raise TryOnError(f"OpenAI API anahtarı geçersiz: {exc}") from exc
                if "model" in msg.lower() and ("404" in msg or "not found" in msg.lower() or "access" in msg.lower()):
                    raise TryOnError(
                        f"gpt-image-1 modeline erişim yok: {exc}\n"
                        "Bu model Tier-1+ hesap gerektiriyor. OpenAI dashboard'dan kota durumunu kontrol edin."
                    ) from exc
                errors.append(f"Deneme {attempt + 1} başarısız: {exc}")

        if not candidates:
            raise TryOnError(
                "Hiçbir örnekleme başarılı olmadı. " + "; ".join(errors)
            )

        # En iyi skora sahip adayı seç
        best_idx = int(np.argmax(scores))
        best_arr = candidates[best_idx]

        warnings: list[str] = []
        if errors:
            warnings.extend(errors)
        if len(candidates) > 1:
            score_summary = ", ".join(f"{s:.2f}" for s in scores)
            warnings.append(
                f"Reinforcement: {len(candidates)}/{num_samples} örnek üretildi. "
                f"Skorlar: [{score_summary}] → Deneme {best_idx + 1} seçildi."
            )

        return TryOnResult(
            composite_rgb=best_arr,
            person_name=request.person_name,
            garment_name=request.garment_name,
            warnings=warnings,
            debug={
                "backend": self.name,
                "model": self.model,
                "quality": quality,
                "size": size,
                "num_samples": num_samples,
                "scores": scores,
                "best_idx": best_idx,
                "color_hint": color_hint,
            },
        )

    def _decode_response(self, response) -> bytes:
        """API yanıtından ham PNG/bytes döndürür."""
        try:
            img_data = response.data[0]
            if getattr(img_data, "b64_json", None):
                return base64.b64decode(img_data.b64_json)
            elif getattr(img_data, "url", None):
                import urllib.request
                with urllib.request.urlopen(img_data.url) as resp:  # noqa: S310
                    return resp.read()
            else:
                raise TryOnError("API yanıtı görüntü içermiyor.")
        except TryOnError:
            raise
        except Exception as exc:
            raise TryOnError(f"Sonuç decode hatası: {exc}") from exc

    @staticmethod
    def _default_prompt(garment_name: str, color_hint: str = "") -> str:
        name_hint = garment_name.replace("-", " ").replace("_", " ") if garment_name else "giysi"
        color_line = (
            f" The garment's dominant colors are {color_hint}."
            if color_hint
            else ""
        )
        return (
            f"Virtual try-on: dress the person in the first image with the '{name_hint}' "
            f"garment shown in the second image.{color_line} "
            "Requirements: "
            "(1) Keep the person's face, hair, skin tone, hands, and body shape identical. "
            "(2) Keep the background and environment completely unchanged. "
            "(3) Replace ONLY the upper body clothing in the masked torso region. "
            "(4) Match the garment's exact color, texture, pattern, and design details. "
            "(5) Simulate realistic fabric draping, wrinkles, and fit as if the person is "
            "actually wearing the garment. "
            "(6) Maintain consistent, natural lighting across the entire image. "
            "The final image must look like a professional fashion photograph."
        )
