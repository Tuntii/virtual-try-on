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
        model: str = "dall-e-2",
        num_samples: int = 2,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.num_samples = max(1, int(num_samples))
        self._http: Any | None = None

    def _resolve_key(self) -> None:
        """Birden fazla kaynaktan API key okur."""
        if self._api_key:
            return
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        if self._api_key:
            return
        try:
            import tomllib
            toml_path = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", ".streamlit", "secrets.toml"
            ))
            with open(toml_path, "rb") as f:
                self._api_key = tomllib.load(f).get("OPENAI_API_KEY", "")
        except Exception:
            pass
        if not self._api_key:
            raise TryOnError(
                "OpenAI API anahtarı bulunamadı. "
                ".streamlit/secrets.toml veya OPENAI_API_KEY env değişkenini kontrol edin."
            )

    def _ensure_http(self) -> None:
        if self._http is not None:
            return
        self._resolve_key()
        try:
            import httpx  # type: ignore
        except ImportError as exc:
            raise TryOnError("httpx kurulu değil. `pip install httpx` ile kurun.") from exc
        # 120s timeout — büyük görüntü yükleme + inference süresi için
        self._http = httpx.Client(
            timeout=httpx.Timeout(120.0, connect=30.0),
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    def _call_edit_api(self, person_png: bytes, mask_png: bytes, prompt: str) -> bytes:
        """httpx ile doğrudan /v1/images/edits çağrısı yapar.

        openai SDK yerine doğrudan HTTP kullanılır çünkü SDK
        gpt-image-1 gibi yeni model isimlerini yerel olarak reddeder.
        """
        import httpx  # type: ignore

        files = {
            "image": ("person.png", io.BytesIO(person_png), "image/png"),
            "mask":  ("mask.png",   io.BytesIO(mask_png),   "image/png"),
        }
        data = {
            "model": self.model,
            "prompt": prompt,
            "n": "1",
            "size": "1024x1024",
        }
        try:
            resp = self._http.post(
                "https://api.openai.com/v1/images/edits",
                files=files,
                data=data,
            )
        except httpx.ConnectError as exc:
            raise TryOnError(
                f"OpenAI bağlantı hatası: {exc}\n"
                "API sunucusuna ulaşılamıyor."
            ) from exc

        if resp.status_code != 200:
            try:
                detail = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                detail = resp.text
            if resp.status_code == 401:
                raise TryOnError(f"OpenAI API anahtarı geçersiz: {detail}")
            raise TryOnError(f"OpenAI API hatası ({resp.status_code}): {detail}")

        img_data = resp.json()["data"][0]
        if img_data.get("b64_json"):
            return base64.b64decode(img_data["b64_json"])
        # dall-e-2 varsayılan olarak URL döndürür
        import urllib.request
        with urllib.request.urlopen(img_data["url"]) as r:  # noqa: S310
            return r.read()

    # ------------------------------------------------------------------
    def run(self, request: TryOnRequest) -> TryOnResult:
        self._ensure_http()

        person_rgb = request.person_rgb
        options = request.options or {}
        num_samples = int(options.get("num_samples", self.num_samples))

        # 1) Torso maskesi
        _silhouette, torso = build_torso_mask(person_rgb)

        # 2) Kişi PNG — max 1024px
        person_png = _square_pad_rgb_png(_to_png_bytes(_resize_max(person_rgb, 1024)))

        # 3) Mask PNG (şeffaf = düzenle = torso alanı)
        torso_resized = _resize_max_gray(torso, 1024, person_rgb.shape[:2])
        mask_png = _square_pad_png(_build_mask_png(torso_resized))

        # 4) Garment renk analizi (prompt için)
        garment = ensure_garment_rgba(request.garment_rgba)
        garment = crop_to_alpha(garment)
        g_rgb = garment[:, :, :3].astype(np.float32)
        g_alpha = garment[:, :, 3:4].astype(np.float32) / 255.0
        garment_rgb = (g_rgb * g_alpha + 255.0 * (1.0 - g_alpha)).clip(0, 255).astype(np.uint8)

        garment_centers = _dominant_kmeans(garment_rgb, k=3)
        color_hint = _dominant_color_desc(garment_rgb, top_n=2)
        prompt = options.get("prompt", self._default_prompt(request.garment_name, color_hint))

        # 5) Best-of-N örnekleme
        candidates: list[np.ndarray] = []
        scores: list[float] = []
        errors: list[str] = []

        for attempt in range(num_samples):
            try:
                raw = self._call_edit_api(person_png, mask_png, prompt)
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
                "num_samples": num_samples,
                "scores": scores,
                "best_idx": best_idx,
                "color_hint": color_hint,
            },
        )

    @staticmethod
    def _default_prompt(garment_name: str, color_hint: str = "") -> str:
        name_hint = garment_name.replace("-", " ").replace("_", " ") if garment_name else "giysi"
        color_line = f" Colors: {color_hint}." if color_hint else ""
        return (
            f"Fashion photo: the person is wearing a '{name_hint}' garment.{color_line} "
            "Keep the person's face, hair, skin, hands, pose, and background exactly the same. "
            "Replace ONLY the clothing on the upper body in the masked area with the garment. "
            "Realistic fabric draping, natural lighting, professional fashion photograph look."
        )
