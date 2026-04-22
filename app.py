"""Virtual Try-On POC — Streamlit uygulaması.

gpt-image-1 ile gerçek zamanlı sanal kiyafet deneme.
Tek ürün + çoklu manken desteği.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.backends.base import TryOnBackend
from src.core.errors import TryOnError
from src.core.types import TryOnRequest, TryOnResult
from src.services import catalog
from src.services.image_io import load_garment_rgba, load_person_rgb
from src.services.quality_gate import check_garment, check_person
from src.ui.components import results_gallery

st.set_page_config(
    page_title="Virtual Try-On POC",
    page_icon="👕",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _cpu_backend():
    from src.backends.cpu_composite_backend import CpuCompositeBackend
    return CpuCompositeBackend()


def _openai_backend(api_key: str):
    from src.backends.openai_image_backend import OpenAiImageBackend
    return OpenAiImageBackend(api_key=api_key)


def _get_api_key() -> str:
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        import os
        return os.environ.get("OPENAI_API_KEY", "")


@st.cache_data(show_spinner=False)
def _cached_person(bytes_: bytes):
    return load_person_rgb(bytes_)


@st.cache_data(show_spinner=False)
def _cached_garment(bytes_: bytes):
    return load_garment_rgba(bytes_)


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _uploaded_to_bytes(file) -> bytes:
    return file.getvalue() if hasattr(file, "getvalue") else file.read()


def _sidebar_controls() -> dict:
    st.sidebar.header("Hakkında")
    st.sidebar.info(
        "ℹ️ Bu bir **Proof of Concept** uygulamasıdır.\n\n"
        "Local ve cloud sistemlerinde çalışabiliyor. Prompt iyileştirmeleri yapılmalı, kalite kontrolü için Product Person fit gerekiyor, feedback için denemekten çekinmeyin."
    )
    st.sidebar.divider()
    show_debug = st.sidebar.checkbox("Debug bilgisi göster", value=False)
    return {"show_debug": show_debug}


def _garment_section() -> tuple[bytes | None, str]:
    st.subheader("1) Ürün görseli")
    samples = catalog.list_garments()
    tab_up, tab_sample = st.tabs(["Yükle", "Örneklerden seç"])
    with tab_up:
        up = st.file_uploader(
            "Ürün (PNG + şeffaf arka plan idealdir)",
            type=["png", "jpg", "jpeg", "webp"],
            key="garment_up",
        )
        if up is not None:
            return _uploaded_to_bytes(up), Path(up.name).stem
    with tab_sample:
        if not samples:
            st.info("assets/samples/garments klasörüne örnek ekleyebilirsiniz.")
        else:
            options = {s.name: s for s in samples}
            pick = st.selectbox("Örnek ürün", list(options.keys()), key="garment_pick")
            if pick:
                item = options[pick]
                st.image(str(item.path), width=220)
                return _read_bytes(item.path), item.name
    return None, ""


def _models_section() -> list[tuple[bytes, str]]:
    st.subheader("2) Manken fotoğrafları")
    results: list[tuple[bytes, str]] = []
    samples = catalog.list_models()
    tab_up, tab_sample = st.tabs(["Yükle (çoklu)", "Örneklerden seç"])
    with tab_up:
        ups = st.file_uploader(
            "Bir veya birden fazla manken fotoğrafı",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="models_up",
        )
        for u in ups or []:
            results.append((_uploaded_to_bytes(u), Path(u.name).stem))
    with tab_sample:
        if not samples:
            st.info("assets/samples/models klasörüne örnek ekleyebilirsiniz.")
        else:
            picks = st.multiselect(
                "Örnek mankenler",
                options=[s.name for s in samples],
                key="models_pick",
            )
            sample_map = {s.name: s for s in samples}
            for name in picks:
                item = sample_map[name]
                results.append((_read_bytes(item.path), item.name))
    return results


def _get_backend(opts: dict) -> TryOnBackend:
    return _openai_backend(_get_api_key())


def _build_options(opts: dict) -> dict:
    return {}


def _run_batch(
    garment_bytes: bytes,
    garment_name: str,
    models: list[tuple[bytes, str]],
    opts: dict,
) -> list[TryOnResult]:
    backend = _get_backend(opts)
    garment_rgba = _cached_garment(garment_bytes)
    backend_opts = _build_options(opts)

    results: list[TryOnResult] = []
    progress = st.progress(0.0, text="Hazırlanıyor…")
    total = len(models)
    for i, (person_bytes, person_name) in enumerate(models, start=1):
        progress.progress((i - 1) / total, text=f"İşleniyor: {person_name}")
        try:
            person_rgb = _cached_person(person_bytes)
            req = TryOnRequest(
                garment_rgba=garment_rgba,
                person_rgb=person_rgb,
                garment_name=garment_name,
                person_name=person_name,
                options=backend_opts,
            )
            results.append(backend.run(req))
        except TryOnError as exc:
            st.error(f"{person_name}: {exc}")
        except Exception as exc:  # pragma: no cover
            st.error(f"{person_name}: beklenmeyen hata — {exc}")
    progress.progress(1.0, text="Tamamlandı")
    progress.empty()
    return results


def _show_quality_warnings(
    garment_bytes: bytes | None, models: list[tuple[bytes, str]]
) -> None:
    if garment_bytes is None and not models:
        return
    warnings: list[str] = []
    if garment_bytes is not None:
        try:
            warnings.extend(check_garment(_cached_garment(garment_bytes)))
        except Exception as exc:
            warnings.append(f"Ürün kontrolü başarısız: {exc}")
    for person_bytes, person_name in models:
        try:
            person_warns = check_person(_cached_person(person_bytes))
            warnings.extend([f"{person_name}: {w}" for w in person_warns])
        except Exception as exc:
            warnings.append(f"{person_name}: {exc}")
    if warnings:
        with st.expander("⚠️ Kalite uyarıları", expanded=True):
            for w in warnings:
                st.warning(w)


def main() -> None:
    st.title("👕 Virtual Try-On")
    st.caption(
        "**Proof of Concept** — Ürün görselini manken fotoğrafına gpt-image-1 ile giydiriyor."
    )

    opts = _sidebar_controls()

    left, right = st.columns([1, 1])
    with left:
        garment_bytes, garment_name = _garment_section()
    with right:
        models = _models_section()

    st.divider()
    _show_quality_warnings(garment_bytes, models)

    ready = garment_bytes is not None and len(models) > 0
    run = st.button(
        "Denemeyi çalıştır",
        type="primary",
        disabled=not ready,
        width="stretch",
    )

    if not ready:
        st.info("Bir ürün ve en az bir manken fotoğrafı seçin.")
        return

    if run:
        st.session_state["_results"] = _run_batch(
            garment_bytes, garment_name or "garment", models, opts
        )

    results = st.session_state.get("_results", [])
    if results:
        st.subheader("3) Sonuçlar")
        results_gallery(results, show_debug=opts["show_debug"])


if __name__ == "__main__":
    main()
