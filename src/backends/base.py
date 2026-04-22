"""Backend arayüzü.

Tüm try-on backend'leri (CPU compositing, ileride GPU/generative) aynı
`run` sözleşmesini uygular. UI katmanı yalnızca bu arayüze bakar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.types import TryOnRequest, TryOnResult


class TryOnBackend(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, request: TryOnRequest) -> TryOnResult:
        """Tek bir ürün + manken çiftini işleyip sonuç döndürür."""
