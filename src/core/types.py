"""Veri sözleşmeleri.

`TryOnRequest` UI katmanından backend'e geçirilen girdiyi, `TryOnResult` ise
backend'ten UI'a dönen çıktıyı tanımlar. Görseller numpy RGB (HxWx3, uint8)
olarak taşınır; alfa kanalı varsa RGBA (HxWx4) olarak taşınabilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class TryOnRequest:
    garment_rgba: np.ndarray  # HxWx4 uint8 (alfa kanalı kritik)
    person_rgb: np.ndarray  # HxWx3 uint8
    garment_name: str = "garment"
    person_name: str = "person"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class TryOnResult:
    composite_rgb: np.ndarray  # HxWx3 uint8 nihai görsel
    debug: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    person_name: str = "person"
    garment_name: str = "garment"
    preview_mask: Optional[np.ndarray] = None  # HxW uint8 (yerleşim maskesi)
