"""Örnek ürün ve manken görsellerini listeler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GARMENTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "samples" / "garments"
MODELS_DIR = Path(__file__).resolve().parents[2] / "assets" / "samples" / "models"

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class SampleItem:
    name: str
    path: Path


def _list(dir_: Path) -> list[SampleItem]:
    if not dir_.exists():
        return []
    items = [
        SampleItem(name=p.stem, path=p)
        for p in sorted(dir_.iterdir())
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]
    return items


def list_garments() -> list[SampleItem]:
    return _list(GARMENTS_DIR)


def list_models() -> list[SampleItem]:
    return _list(MODELS_DIR)
