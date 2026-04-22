"""Kullanıcı dostu hata tipleri.

UI katmanı bu tiplerden yakalayıp `st.error` ile kısa bir mesaj gösterir.
Beklenmeyen hatalar için `TryOnError` genel tipi kullanılır.
"""

from __future__ import annotations


class TryOnError(Exception):
    """POC genel hata tipi."""


class InvalidImageError(TryOnError):
    """Bozuk veya desteklenmeyen görsel."""


class PersonNotDetectedError(TryOnError):
    """Manken fotoğrafında insan/omuz landmark'ı tespit edilemedi."""


class GarmentMaskError(TryOnError):
    """Ürün maskesi üretilemedi."""
