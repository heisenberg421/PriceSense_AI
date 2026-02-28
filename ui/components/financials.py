"""
ui/components/financials.py
----------------------------
Catalog impact table and detailed financials expander.
"""

from __future__ import annotations
import pandas as pd
import streamlit as st


def render_catalog_impact(r: dict) -> None:
    """
    Toggle-able table showing per-SKU cannibalization bleed / halo effect.
    Only rendered when sku_breakdown is present in the result.
    """
    if not r.get("sku_breakdown"):
        return

    st.divider()
    col_head, col_toggle = st.columns([2, 1])
    col_head.markdown("**Catalog Impact Detail**")
    show_all = col_toggle.toggle(
        "Show all SKUs", value=False,
        help="Toggle to include unrelated SKUs with zero impact",
    )

    rows = []
    for s in r["sku_breakdown"]:
        if not show_all and s["relationship"] == "unrelated" and s["lost_units"] == 0:
            continue

        if s["lost_units"] == 0:
            effect_label, unit_display = "○ Neutral", "0"
        else:
            effect_label = "▲ Halo" if s["direction"] == "gain" else "▼ Bleed"
            unit_display = f"{'−' if s['direction'] == 'loss' else '+'}{s['lost_units']}"

        rows.append({
            "SKU":     s["id"],
            "Product": s["name"],
            "Rel.":    s["relationship"].capitalize(),
            "Effect":  effect_label,
            "Units":   unit_display,
            "Rev. $":  f"${abs(s['revenue_impact']):,.0f}" if s["revenue_impact"] != 0 else "$0",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_detailed_financials(r: dict) -> None:
    """
    Collapsible expander showing baseline vs promo revenue/profit,
    plus deduction breakdown (cannibalization + post-promo hangover).
    """
    with st.expander("📊 Detailed Financials"):
        fa, fb = st.columns(2)
        fa.markdown("**Baseline**")
        fa.metric("Revenue", f"${r['baseline_revenue']:,.0f}")
        fa.metric("Profit",  f"${r['baseline_profit']:,.0f}")
        fb.markdown("**With Promo**")
        fb.metric("Revenue", f"${r['promo_revenue']:,.0f}")
        fb.metric("Profit",  f"${r['promo_profit']:,.0f}")

        dip_loss  = r.get("post_promo_dip_loss", 0.0)
        cann_loss = r.get("total_cannibalized_revenue", 0.0)
        if dip_loss > 0 or cann_loss > 0:
            st.markdown("**Deductions from Net Profit**")
            dl1, dl2 = st.columns(2)
            dl1.metric("Cannibalization Loss", f"−${cann_loss:,.0f}")
            dl2.metric("Post-Promo Hangover",  f"−${dip_loss:,.0f}")
