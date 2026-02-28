"""
ui/screen_main.py
-----------------
Single-page side-by-side layout.
Left:  data upload + catalog + question input.
Right: analysis results.

This file only owns the page layout and the run_engine call.
All rendering logic lives in ui/components/.
All data helpers live in ui/utils/.
"""

from __future__ import annotations
import os

import pandas as pd
import streamlit as st

from engine.orchestrator        import run_engine
from engine.relationship        import classify_relationships
from ui.utils.catalog           import build_catalog_from_df
from ui.utils.intent_parser     import parse_question
from ui.components.result_panel import render_result


def render_home() -> None:
    from config.settings import get_openai_api_key
    api_key = get_openai_api_key() or st.session_state.get("api_key", "")

    left, right = st.columns([1, 1], gap="large")

    # ── LEFT PANEL ────────────────────────────────────────────────────────────
    with left:
        hist_df: pd.DataFrame | None = st.session_state.get("historical_df")

        # Try bundled data first (silent)
        if hist_df is None:
            try:
                default = os.path.join(
                    os.path.dirname(__file__), "..", "data", "extended_historical_data.csv"
                )
                if os.path.exists(default):
                    hist_df = pd.read_csv(default)
                    st.session_state["historical_df"] = hist_df
            except Exception:
                pass

        # Show uploader when no data is loaded yet
        if hist_df is None:
            st.markdown("### 📂 Upload Sales Data")
            st.caption(
                "CSV columns: `sku`, `category`, `week`, `price`, `cost`, "
                "`promo_flag`, `discount`, `units_sold`"
            )
            uploaded = st.file_uploader(
                "Upload CSV", type=["csv"], key="hist_uploader", label_visibility="collapsed"
            )
            if uploaded is not None:
                try:
                    df = pd.read_csv(uploaded)
                    st.session_state["historical_df"] = df
                    st.session_state.pop("catalog_skus", None)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to parse: {exc}")
            return  # nothing else to show until data is loaded

        # Optional uploader to swap data
        with st.expander("🔄 Replace data"):
            st.caption(
                "CSV columns: `sku`, `category`, `week`, `price`, `cost`, "
                "`promo_flag`, `discount`, `units_sold`"
            )
            uploaded = st.file_uploader(
                "Upload new CSV", type=["csv"], key="hist_uploader", label_visibility="collapsed"
            )
            if uploaded is not None:
                try:
                    df = pd.read_csv(uploaded)
                    st.session_state["historical_df"] = df
                    st.session_state.pop("catalog_skus", None)
                    st.session_state.pop("result", None)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to parse: {exc}")

        # ── Catalog ───────────────────────────────────────────────────────────
        if "catalog_skus" not in st.session_state or not st.session_state["catalog_skus"]:
            st.session_state["catalog_skus"] = build_catalog_from_df(hist_df)

        catalog_skus: list[dict] = st.session_state["catalog_skus"]

        st.markdown("### 🗂️ Product Catalog")
        st.dataframe(
            pd.DataFrame([{
                "SKU":     s["id"],
                "Product": s["name"],
                "Price":   f"${s['price']:.2f}",
                "Margin":  f"{s['margin']:.1f}%",
            } for s in catalog_skus]),
            use_container_width=True,
            hide_index=True,
        )

        # ── Question ──────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 💬 What promotion are you considering?")
        st.caption('e.g. "Should I run 25% off on Salted Pistachios 16oz next week?"')

        question = st.text_input(
            "question",
            value=st.session_state.get("promotion_question", ""),
            placeholder="e.g. Should we run 25% off on Salted Pistachios 16oz next week?",
            key="promotion_question_input",
            label_visibility="collapsed",
        )
        st.session_state["promotion_question"] = question

        if st.button(
            "🚀 Analyse Promotion",
            type="primary",
            disabled=not question.strip(),
            use_container_width=True,
            key="btn_run",
        ):
            with st.spinner("Running analysis…"):
                try:
                    parsed = parse_question(question, catalog_skus, api_key)

                    valid_ids = [s["id"] for s in catalog_skus]
                    sku_id    = parsed.get("sku_id") or (valid_ids[0] if valid_ids else "")
                    if sku_id not in valid_ids:
                        sku_id = valid_ids[0] if valid_ids else ""

                    discount_pct = max(5, min(50, int(parsed.get("discount_pct") or 25)))

                    updated = [dict(s) for s in catalog_skus]
                    rels    = classify_relationships(updated, sku_id)
                    for sku, rel in zip(updated, rels):
                        sku["relationship"] = rel

                    promoted = next((s for s in updated if s["id"] == sku_id), updated[0])

                    result = run_engine({
                        "api_key":       api_key,
                        "retailer_name": "Retailer",
                        "historical_df": hist_df,
                        "promoted_sku":  promoted,
                        "catalog_skus":  updated,
                        "discount_pct":  discount_pct,
                        "duration_days": 7,
                        "context_text":  "",
                    })
                    st.session_state["result"] = result
                    st.rerun()

                except Exception as exc:
                    st.error(f"**Analysis failed:** {exc}")

    # ── RIGHT PANEL ───────────────────────────────────────────────────────────
    with right:
        result = st.session_state.get("result")

        if result is None:
            st.markdown("### 📊 Analysis Results")
            st.markdown(
                "<div style='height:300px; display:flex; align-items:center;"
                " justify-content:center; color:#888; border: 1px dashed #ccc;"
                " border-radius:8px; text-align:center; padding:2rem'>"
                "<div><div style='font-size:2.5rem'>📈</div>"
                "<div style='margin-top:0.75rem; font-size:0.95rem'>"
                "Results will appear here<br>after you run an analysis.</div></div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("### 📊 Analysis Results")
            render_result(result)
            if st.button("🔄 Clear", key="btn_clear"):
                st.session_state.pop("result", None)
                st.session_state.pop("promotion_question", None)
                st.rerun()
