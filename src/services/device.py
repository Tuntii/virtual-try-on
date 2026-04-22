"""Cihaz (CUDA/CPU) yardımcıları."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceInfo:
    torch_available: bool
    cuda_available: bool
    device: str  # "cuda" | "cpu" | "unavailable"
    name: str = ""
    total_vram_gb: float = 0.0


def probe_device() -> DeviceInfo:
    try:
        import torch  # type: ignore
    except Exception:
        return DeviceInfo(torch_available=False, cuda_available=False, device="unavailable")

    if not torch.cuda.is_available():
        return DeviceInfo(torch_available=True, cuda_available=False, device="cpu")

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return DeviceInfo(
        torch_available=True,
        cuda_available=True,
        device="cuda",
        name=props.name,
        total_vram_gb=props.total_memory / (1024**3),
    )
