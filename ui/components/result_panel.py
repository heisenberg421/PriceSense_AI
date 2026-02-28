"""
ui/components/result_panel.py
------------------------------
Top-level result renderer. Assembles all sub-components into the right panel.

Layout
------
  Verdict banner + fatigue warning (if fired)
  ├── Core metrics (2×2 grid)
  ├── Signal insights
  │     trend → seasonality → recency → SNR → post-promo dip
  ├── Risk breakdown bar (4 or 5 tiles)
  │     competitor price detail
  │     promo fatigue
  ├── Analyst brief
  ├── Catalog impact table
  └── Detailed financials expander
"""

from __future__ import annotations
import streamlit as st

from ui.components.signals import (
    render_trend_signal,
    render_recency_signal,
    render_snr_signal,
    render_seasonality_signal,
    render_post_promo_dip,
    render_competitor_price,
    render_promo_fatigue,
)
from ui.components.financials import render_catalog_impact, render_detailed_financials


_RISK_EMOJI  = {"low": "🟢", "medium": "🟡", "high": "🔴"}
_CONF_COLORS = {"high": "green", "medium": "orange", "low": "red"}
_METHOD_LABELS = {
    "observed":         "📊 Observed historical promotions",
    "elasticity_model": "📈 Price elasticity model",
    "fallback":         "⚠️ Estimated (limited promo history)",
}


def render_result(r: dict) -> None:
    """Render the full analysis result for one promotion."""

    # ── Verdict ───────────────────────────────────────────────────────────────
    getattr(st, r["streamlit_type"])(f"{r['icon']} &nbsp; **{r['label']}** — {r['summary']}")
    if r.get("suggested_change"):
        st.info(f"**Suggested adjustment:** {r['suggested_change']}")

    method = r.get("lift_method", "model")
    conf   = r.get("lift_confidence", "medium")
    st.caption(
        f"{_METHOD_LABELS.get(method, 'Model estimate')} · "
        f"Confidence: **:{_CONF_COLORS[conf]}[{conf.upper()}]**"
    )

    st.divider()

    # ── Core metrics (2×2) ────────────────────────────────────────────────────
    mc1, mc2 = st.columns(2)
    mc1.metric("Unit Lift",       f"+{r['lift_pct']}%",        f"+{r['incremental_units']} units")
    mc2.metric("Cannibalization", f"{r['cannibalized_pct']}%", "of lift lost")

    mc3, mc4 = st.columns(2)
    mc3.metric("Revenue Uplift", f"${r['revenue_uplift']:,.0f}", "vs. baseline", delta_color="off")

    profit = r["net_profit_delta"]
    mc4.metric(
        "Net Profit Impact",
        f"{'-' if profit < 0 else ''}${abs(profit):,.0f}",
        "after all deductions",
        delta_color="normal" if profit >= 0 else "inverse",
    )

    # ── Signal insights ───────────────────────────────────────────────────────
    render_trend_signal(r)
    render_seasonality_signal(r)
    render_recency_signal(r)
    render_snr_signal(r)
    render_post_promo_dip(r)

    st.divider()

    # ── Risk breakdown ────────────────────────────────────────────────────────
    st.markdown("**Risk Breakdown**")
    comp_risk = r.get("competitor_price_risk")

    risk_tiles = [
        ("Margin",    r["margin_risk"]),
        ("Cannibal.", r["cannibalization_risk"]),
        ("Timing",    r["timing_risk"]),
    ]
    if comp_risk:
        risk_tiles.append(("Competitor", comp_risk))
    risk_tiles.append(("Overall", r["overall_risk"]))

    for col, (label, level) in zip(st.columns(len(risk_tiles)), risk_tiles):
        col.metric(label, _RISK_EMOJI[level])
        col.caption(f"**{level.upper()}**")

    render_competitor_price(r)
    render_promo_fatigue(r)

    st.divider()

    # ── Analyst brief ─────────────────────────────────────────────────────────
    st.markdown("**Analyst Brief**")
    with st.container(border=True):
        for para in [p.strip() for p in r["narrative"].split("\n\n") if p.strip()]:
            st.write(para)

    # ── Catalog impact + financials ───────────────────────────────────────────
    render_catalog_impact(r)
    render_detailed_financials(r)
