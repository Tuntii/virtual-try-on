"""Streamlit UI bileşenleri."""

from __future__ import annotations

from typing import Iterable

import streamlit as st

from src.core.types import TryOnResult
from src.services.image_io import encode_png


def result_card(result: TryOnResult, show_debug: bool = False) -> None:
    with st.container(border=True):
        st.markdown(f"**{result.person_name}** × _{result.garment_name}_")
        st.image(result.composite_rgb, use_container_width=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button(
                "PNG indir",
                data=encode_png(result.composite_rgb),
                file_name=f"tryon_{result.person_name}_{result.garment_name}.png",
                mime="image/png",
                key=f"dl_{result.person_name}_{result.garment_name}",
            )
        with col2:
            if show_debug and result.preview_mask is not None:
                with st.popover("Debug"):
                    st.json(result.debug)
                    st.image(result.preview_mask, caption="Yerleşim maskesi")
        for w in result.warnings:
            st.warning(w)


def results_gallery(results: Iterable[TryOnResult], show_debug: bool = False) -> None:
    results = list(results)
    if not results:
        return
    cols = st.columns(2)
    for i, r in enumerate(results):
        with cols[i % 2]:
            result_card(r, show_debug=show_debug)
