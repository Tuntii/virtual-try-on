"""GPU tabanlı generative backend.

Stable Diffusion Inpainting + IP-Adapter ile torso bölgesini, ürün
görselinin stilini referans alarak yeniden üretir. Düşük VRAM'li
GPU'lar (6 GB) için fp16 ve attention/VAE slicing ile çalışacak
şekilde yapılandırılır.

Not: Bu backend `torch` ve `diffusers` paketlerini gerektirir
(requirements-gpu.txt). Modüller yalnızca backend ilk kez seçildiğinde
içe aktarılır.
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.backends.base import TryOnBackend
from src.core.errors import TryOnError
from src.core.types import TryOnRequest, TryOnResult
from src.services.garment_mask import crop_to_alpha, ensure_garment_rgba
from src.services.torso_mask import build_torso_mask


DEFAULT_PROMPT = (
    "a person wearing stylish clothing, studio fashion photo, "
    "photorealistic, sharp focus, natural lighting, detailed fabric"
)
DEFAULT_NEGATIVE = (
    "blurry, lowres, bad anatomy, extra limbs, disfigured, watermark, text, "
    "deformed hands, naked, nsfw"
)


def _to_pil_rgb(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _fit_multiple_of_8(w: int, h: int, target_short: int) -> tuple[int, int]:
    """Kısa kenarı hedefe getirip 8'in katlarına yuvarlar (SD gereksinimi)."""
    scale = target_short / float(min(w, h))
    nw = max(int(round(w * scale / 8)) * 8, 256)
    nh = max(int(round(h * scale / 8)) * 8, 256)
    return nw, nh


class GpuInpaintBackend(TryOnBackend):
    name = "gpu-inpaint"

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        ip_adapter_repo: str = "h94/IP-Adapter",
        # ip-adapter_sd15.bin: SD'nin kendi CLIP encoder'ını kullanır,
        # ayrı ViT-H encoder gerektirmez → 6 GB VRAM'de güvenli.
        # ip-adapter-plus_sd15.bin: ek ViT-H (~2.7 GB) ister, 6 GB'ta OOM riski.
        ip_adapter_weight: str = "ip-adapter_sd15.bin",
        device: str | None = None,
        dtype: str = "fp16",
        target_short_side: int = 512,
    ) -> None:
        self.model_id = model_id
        self.ip_adapter_repo = ip_adapter_repo
        self.ip_adapter_weight = ip_adapter_weight
        self._device_override = device
        self._dtype_pref = dtype
        self.target_short_side = target_short_side
        self._pipe: Any | None = None
        self._torch: Any | None = None

    # -- lazy init --------------------------------------------------------
    def _ensure_pipe(self) -> None:
        if self._pipe is not None:
            return

        try:
            import torch  # type: ignore
            from diffusers import StableDiffusionInpaintPipeline  # type: ignore
        except Exception as exc:
            raise TryOnError(
                "GPU backend için `torch` ve `diffusers` kurulu değil. "
                "`pip install -r requirements-gpu.txt` ile kurun."
            ) from exc

        self._torch = torch

        if self._device_override:
            device = self._device_override
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cpu":
            raise TryOnError(
                "GPU backend CUDA gerektirir ancak CUDA bulunamadı. "
                "NVIDIA sürücüsü ve CUDA-enabled torch kurulumunu doğrulayın."
            )

        dtype = torch.float16 if self._dtype_pref == "fp16" else torch.float32

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        try:
            # image_encoder_folder="models/image_encoder": h94/IP-Adapter repo'sundaki
            # ViT-H-14 CLIP encoder'ı yükler. Her iki variant (standart ve plus)
            # bu encoder'a ihtiyaç duyar; None geçilirse self.image_encoder=None olur
            # ve inference'ta 'NoneType has no attribute parameters' hatası alınır.
            pipe.load_ip_adapter(
                self.ip_adapter_repo,
                subfolder="models",
                weight_name=self.ip_adapter_weight,
                image_encoder_folder="models/image_encoder",
            )
        except Exception as exc:  # pragma: no cover
            raise TryOnError(
                f"IP-Adapter ağırlıkları yüklenemedi: {exc}. "
                "İnternet bağlantısı ve `huggingface_hub` erişimini doğrulayın."
            ) from exc

        # Toplam model boyutu ~3.8 GB (fp16) → 6 GB VRAM'e sığır.
        # enable_model_cpu_offload + IP-Adapter encode interaksiyonu
        # diffusers 0.27+'da 'tuple has no attribute shape' hatasına yol açar;
        # doğrudan pipe.to(device) kullanmak daha güvenli.
        pipe.to(device)
        # Bellek optimizasyonları (cpu_offload olmadan)
        try:
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        except Exception:
            pass

        self._pipe = pipe

    # -- public API -------------------------------------------------------
    def run(self, request: TryOnRequest) -> TryOnResult:
        self._ensure_pipe()
        options = request.options or {}
        strength = float(options.get("strength", 0.98))
        guidance_scale = float(options.get("guidance_scale", 7.0))
        num_steps = int(options.get("num_steps", 30))
        ip_scale = float(options.get("ip_scale", 0.9))
        prompt = options.get("prompt", DEFAULT_PROMPT)
        negative = options.get("negative_prompt", DEFAULT_NEGATIVE)
        seed = options.get("seed")

        person_rgb = request.person_rgb
        h, w = person_rgb.shape[:2]

        # 1) Torso inpaint maskesi
        _silhouette, torso = build_torso_mask(person_rgb)

        # 2) Ürün görseli (IP-Adapter referansı)
        garment = ensure_garment_rgba(request.garment_rgba)
        garment = crop_to_alpha(garment)
        # Beyaz zemine bastır (IP-Adapter için RGB lazım)
        g_rgb = garment[:, :, :3].astype(np.float32)
        g_alpha = garment[:, :, 3:4].astype(np.float32) / 255.0
        garment_rgb = (g_rgb * g_alpha + 255.0 * (1.0 - g_alpha)).clip(0, 255).astype(np.uint8)

        # 3) SD için 8'in katlarına yuvarla
        tw, th = _fit_multiple_of_8(w, h, self.target_short_side)
        person_resized = cv2.resize(person_rgb, (tw, th), interpolation=cv2.INTER_AREA)
        torso_resized = cv2.resize(torso, (tw, th), interpolation=cv2.INTER_LINEAR)

        person_pil = _to_pil_rgb(person_resized)
        mask_pil = Image.fromarray(torso_resized, mode="L")
        garment_pil = _to_pil_rgb(garment_rgb)

        # 4) IP-Adapter scale
        try:
            self._pipe.set_ip_adapter_scale(ip_scale)
        except Exception:
            pass

        generator = None
        if seed is not None:
            generator = self._torch.Generator(device="cuda").manual_seed(int(seed))

        # 5) IP-Adapter görsel embedding'ini elle hesapla.
        # Pipe'a ip_adapter_image geçmek diffusers 0.27+'da encode_image()
        # çıktısını tuple olarak döndürüp '.shape' hatasına yol açar.
        # Ön-encode edip ip_adapter_image_embeds geçmek bu kodu tamamen bypass eder.
        try:
            unet_device = next(self._pipe.unet.parameters()).device
            unet_dtype = next(self._pipe.unet.parameters()).dtype

            feat_inputs = self._pipe.feature_extractor(
                images=[garment_pil], return_tensors="pt"
            )
            pv = feat_inputs.pixel_values.to(device=unet_device, dtype=unet_dtype)
            with self._torch.no_grad():
                enc_out = self._pipe.image_encoder(pv)
                pos_embeds = enc_out.image_embeds          # [1, 768]
                neg_embeds = self._torch.zeros_like(pos_embeds)
                # CFG: [uncond; cond] → [2, 768] → seq dim → [2, 1, 768]
                ip_embeds = self._torch.cat([neg_embeds, pos_embeds]).unsqueeze(1)

            ip_adapter_image_embeds = [ip_embeds]
        except Exception as enc_exc:
            raise TryOnError(f"IP-Adapter embedding hatası: {enc_exc}") from enc_exc

        # 6) İnference
        try:
            out = self._pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=person_pil,
                mask_image=mask_pil,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
            ).images[0]
        except Exception as exc:
            raise TryOnError(f"GPU inference başarısız: {exc}") from exc

        # 6) Orijinal çözünürlüğe geri ölçekle
        result_arr = np.asarray(out.convert("RGB"), dtype=np.uint8)
        if (tw, th) != (w, h):
            result_arr = cv2.resize(result_arr, (w, h), interpolation=cv2.INTER_CUBIC)

        debug = {
            "backend": self.name,
            "model": self.model_id,
            "size_used": (tw, th),
            "steps": num_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "ip_scale": ip_scale,
        }

        return TryOnResult(
            composite_rgb=result_arr,
            debug=debug,
            warnings=[],
            person_name=request.person_name,
            garment_name=request.garment_name,
            preview_mask=torso,
        )
