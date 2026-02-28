"""
ui/components/signals.py
-------------------------
Signal renderers for the analysis results panel.

Each renderer is self-contained: it reads from the result dict `r`,
decides whether the signal is worth showing, and renders compactly.
Silent (caption only) when the signal is below threshold or has no data.

Functions
---------
render_trend_signal(r)          — demand trend slope + trend-adjusted baseline
render_recency_signal(r)        — recency-weighted vs simple-avg lift correction
render_snr_signal(r)            — SNR confidence downgrade warning
render_seasonality_signal(r)    — seasonality index + adjusted baseline  [Feature 4]
render_post_promo_dip(r)        — post-promo hangover measurement          [Feature 5]
render_competitor_price(r)      — competitor price gap + timing risk        [Feature 6]
render_promo_fatigue(r)         — fatigue detection per discount bucket     [Feature 7]
"""

from __future__ import annotations
import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Demand trend
# ─────────────────────────────────────────────────────────────────────────────

def render_trend_signal(r: dict) -> None:
    """
    Shows trend direction, slope, R², and the adjustment made to the baseline.
    Only rendered when |adjustment| >= 1 unit AND R² >= 0.05.
    """
    raw      = r.get("baseline_units_raw", 0.0)
    adjusted = r.get("baseline_units", raw)
    slope    = r.get("trend_slope", 0.0)
    r2       = r.get("trend_r2", 0.0)

    delta      = adjusted - raw
    delta_pct  = (delta / raw * 100) if raw else 0.0
    meaningful = abs(delta) >= 1.0 and r2 >= 0.05

    if not meaningful:
        st.caption(
            f"📉 Demand trend: **flat / no clear signal** "
            f"(baseline {raw:.0f} units/wk, R²={r2:.2f})"
        )
        return

    arrow, label, color = ("▲", "growing", "green") if slope > 0 else ("▼", "declining", "red")
    strength = "strong" if r2 >= 0.5 else ("moderate" if r2 >= 0.2 else "weak")

    st.markdown(
        f"**📉 Demand Trend** &nbsp;·&nbsp; "
        f":{color}[{arrow} {label}] at **{abs(slope):.2f} units/wk** "
        f"({strength} fit, R²={r2:.2f})"
    )

    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Historical Mean",         f"{raw:.0f} units/wk",
               help="Flat all-time average of non-promotional weeks")
    tc2.metric("Trend-Adjusted Baseline", f"{adjusted:.0f} units/wk",
               delta=f"{delta:+.0f} ({delta_pct:+.1f}%)",
               delta_color="normal" if delta >= 0 else "inverse",
               help="OLS projection blended with historical mean using R² as weight.")
    tc3.metric("Baseline Used",           "Trend-adjusted",
               help="Which baseline drove the lift and profit calculations")

    st.caption(
        f"ℹ️ Lift and profit are projected from **{adjusted:.0f} units/wk** "
        f"(not the {raw:.0f} flat mean) because this SKU shows a {strength} {label} trend. "
        f"{'A lower baseline reduces projected incremental units.' if delta < 0 else 'A higher baseline increases projected incremental units.'}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Recency-weighted lift
# ─────────────────────────────────────────────────────────────────────────────

def render_recency_signal(r: dict) -> None:
    """
    Shows the recency correction when observed promo data was used AND
    the correction is >= 0.5pp. Silent otherwise.
    """
    weighted = r.get("recency_weighted_lift")
    raw      = r.get("raw_avg_lift")
    effect   = r.get("recency_effect", 0.0)
    method   = r.get("lift_method", "")

    if method != "observed" or weighted is None or raw is None or abs(effect) < 0.5:
        return

    color          = "orange" if effect < 0 else "green"
    arrow          = "▼" if effect < 0 else "▲"
    correction_dir = "downward" if effect < 0 else "upward"

    st.markdown(
        f"**⏱ Recency Correction** &nbsp;·&nbsp; "
        f":{color}[{arrow} {abs(effect):.1f}pp {correction_dir}] "
        f"— recent promos {'underperformed' if effect < 0 else 'outperformed'} historical average"
    )

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Simple Avg Lift",        f"{raw:.1f}%",
               help="Unweighted average — treats a promo from 2 years ago same as last month.")
    rc2.metric("Recency-Weighted Lift",  f"{weighted:.1f}%",
               delta=f"{effect:+.1f}pp",
               delta_color="normal" if effect >= 0 else "inverse",
               help="Exponential decay weighting (half-life ≈ 35 weeks). This is what the engine used.")
    rc3.metric("Half-Life", "~35 weeks",
               help="A promo from 35 weeks ago gets 50% weight of a current promo.")

    st.caption(
        f"ℹ️ The engine used **{weighted:.1f}%** lift (not {raw:.1f}%) because "
        f"promotions run closer to today carry more predictive weight. "
        f"Lift projection is {'lower' if effect < 0 else 'higher'} than a naive average would suggest."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SNR confidence guard
# ─────────────────────────────────────────────────────────────────────────────

def render_snr_signal(r: dict) -> None:
    """
    Shows the SNR confidence downgrade when it fires. Silent when no downgrade.
    """
    conf_before = r.get("confidence_before_snr", r.get("lift_confidence", "medium"))
    conf_after  = r.get("lift_confidence", "medium")
    snr         = r.get("lift_snr", 0.0)
    inc_units   = r.get("incremental_units", 0)
    std_approx  = round(inc_units / snr, 1) if snr > 0 else 0.0

    if conf_before == conf_after:
        snr_label = f"{snr:.1f}×" if snr < 99 else ">99×"
        st.caption(
            f"📶 Demand signal: **SNR {snr_label}** — lift is clearly above "
            f"background noise (confidence unchanged at {conf_after.upper()})"
        )
        return

    st.warning(
        f"⚠️ **Confidence downgraded: {conf_before.upper()} → {conf_after.upper()}** "
        f"because the expected lift ({inc_units} units) is close to normal "
        f"weekly demand noise (~{std_approx:.0f} units std). "
        f"SNR = {snr:.1f}× (threshold: 1.0×).",
        icon=None,
    )

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Expected Lift",      f"{inc_units} units",
               help="Incremental units the promo is expected to generate.")
    sc2.metric("Demand Noise (std)", f"~{std_approx:.0f} units",
               help="Standard deviation of weekly baseline sales — the noise floor.")
    sc3.metric("Signal-to-Noise",    f"{snr:.1f}×",
               delta=f"{'OK' if snr >= 2.0 else 'LOW' if snr < 1.0 else 'BORDERLINE'}",
               delta_color="normal" if snr >= 2.0 else "inverse",
               help="SNR < 1.0 → LOW. 1.0–2.0 → MEDIUM. > 2.0 → no downgrade.")

    st.caption(
        "ℹ️ When SNR is low, realised lift could easily be zero. "
        "Reduce the discount depth or choose a higher-volume week."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4 — Seasonality Index
# ─────────────────────────────────────────────────────────────────────────────

def render_seasonality_signal(r: dict) -> None:
    """
    Shows the seasonality multiplier applied to the baseline.
    Only rendered when the index deviates >= 3% from neutral.
    """
    index    = r.get("seasonality_index", 1.0)
    adj_base = r.get("seasonality_adjusted_baseline", r.get("baseline_units", 0.0))
    raw_base = r.get("baseline_units_raw", adj_base)
    delta_pct = (index - 1.0) * 100

    if abs(delta_pct) < 3.0:
        st.caption(
            f"📅 Seasonality: **neutral week** (index {index:.2f}) — "
            f"no seasonal adjustment applied to baseline"
        )
        return

    is_peak   = index > 1.0
    color     = "green" if is_peak else "red"
    arrow     = "▲" if is_peak else "▼"
    label     = "peak" if is_peak else "trough"
    adj_delta = adj_base - raw_base

    st.markdown(
        f"**📅 Seasonality** &nbsp;·&nbsp; "
        f":{color}[{arrow} {label.upper()} week] &nbsp; index **{index:.2f}×** "
        f"({delta_pct:+.1f}% vs annual average)"
    )

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Trend Baseline",          f"{raw_base:.0f} units/wk",
               help="Trend-adjusted baseline before seasonal scaling.")
    sc2.metric("Seasonality Index",       f"{index:.2f}×",
               delta=f"{delta_pct:+.1f}%",
               delta_color="normal" if is_peak else "inverse",
               help="Ratio of this week-of-year's mean to the annual mean. Non-promo weeks only.")
    sc3.metric("Season-Adjusted Baseline", f"{adj_base:.0f} units/wk",
               delta=f"{adj_delta:+.0f} units",
               delta_color="normal" if is_peak else "inverse",
               help="This is what the engine used to compute incremental units.")

    st.caption(
        f"ℹ️ Because this is a historically **{label}** week, the baseline was "
        f"{'raised' if is_peak else 'lowered'} by {abs(adj_delta):.0f} units before projecting lift. "
        f"{'Peak weeks amplify promo ROI.' if is_peak else 'Trough weeks dampen projected lift.'}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5 — Post-Promo Dip
# ─────────────────────────────────────────────────────────────────────────────

def render_post_promo_dip(r: dict) -> None:
    """
    Shows the post-promo demand hangover deducted from net profit.
    Silent when no hangover detected.
    """
    dip_data  = r.get("post_promo_dip", {})
    dip_w1    = float(dip_data.get("dip_units_week1", 0.0))
    dip_w2    = float(dip_data.get("dip_units_week2", 0.0))
    total_dip = float(dip_data.get("total_dip_units", 0.0))
    dip_obs   = int(dip_data.get("dip_obs_count", 0))
    dip_conf  = dip_data.get("dip_confidence", "low")
    dip_loss  = float(r.get("post_promo_dip_loss", 0.0))

    if total_dip < 0.5:
        st.caption("🛒 Post-promo dip: **none detected** — no pantry-loading hangover in history")
        return

    conf_color = {"high": "green", "medium": "orange", "low": "red"}.get(dip_conf, "orange")

    st.markdown(
        f"**🛒 Post-Promo Hangover** &nbsp;·&nbsp; "
        f":{conf_color}[{dip_conf.upper()} confidence] — "
        f"avg **{total_dip:.0f} units** demand deficit across 2 weeks post-promo"
    )

    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Wk+1 Dip",       f"−{dip_w1:.0f} units",
               help="Avg unit deficit in the first week after the promo ends.")
    dc2.metric("Wk+2 Dip",       f"−{dip_w2:.0f} units",
               help="Avg unit deficit in the second week after the promo ends.")
    dc3.metric("Profit Deducted", f"−${dip_loss:,.0f}",
               help="Profit lost due to hangover, already deducted from net profit delta.")

    st.caption(
        f"ℹ️ Based on {dip_obs} historical promo event(s). Customers who pantry-loaded "
        f"buy less in the following weeks. This ${dip_loss:,.0f} hangover cost is "
        f"already reflected in the net profit figure above."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature 6 — Competitor Price Signal
# ─────────────────────────────────────────────────────────────────────────────

def render_competitor_price(r: dict) -> None:
    """
    Shows the competitor price gap and its effect on timing risk.
    Silent (with hint) when no competitor_price column in CSV.
    """
    comp_price   = r.get("competitor_price")
    comp_risk    = r.get("competitor_price_risk")
    discount_pct = r.get("discount_pct", 0.0)
    disc_price   = r.get("discounted_price", 0.0)

    your_price = round(disc_price / (1 - discount_pct / 100), 2) if discount_pct < 100 else disc_price

    if comp_price is None or comp_risk is None:
        st.caption(
            "🏪 Competitor price: **no data in CSV** — "
            "add a `competitor_price` column to enable this signal"
        )
        return

    gap_pct = (your_price - comp_price) / your_price * 100 if your_price > 0 else 0.0
    color   = {"high": "red", "medium": "orange", "low": "green"}.get(comp_risk, "orange")
    gap_dir = "above" if gap_pct > 0 else "below"
    arrow   = "▲" if gap_pct > 0 else "▼"

    st.markdown(
        f"**🏪 Competitor Price** &nbsp;·&nbsp; "
        f":{color}[{arrow} {abs(gap_pct):.1f}% {gap_dir} competitor] — "
        f"timing risk: :{color}[**{comp_risk.upper()}**]"
    )

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Your Full Price",    f"${your_price:.2f}",
               help="Historical mean price in non-promotional weeks.")
    cc2.metric("Competitor Price",   f"${comp_price:.2f}",
               help="Mean competitor_price from the historical CSV.")
    cc3.metric("Price Gap",          f"{gap_pct:+.1f}%",
               delta=f"{'Higher risk' if gap_pct > 10 else 'Lower risk' if gap_pct < -5 else 'Neutral'}",
               delta_color="inverse" if gap_pct > 10 else ("normal" if gap_pct < -5 else "off"),
               help=">+10% → significantly more expensive → HIGH risk. <−5% → already cheaper → LOW risk.")

    if comp_risk == "high":
        st.caption(
            f"ℹ️ Your price is ${your_price - comp_price:+.2f} vs competitor. "
            f"Customers may already be choosing the cheaper option."
        )
    elif comp_risk == "low":
        st.caption(
            f"ℹ️ You're already undercutting the competitor by ${comp_price - your_price:.2f}. "
            f"The promo may be unnecessary from a competitive timing standpoint."
        )
    else:
        st.caption(
            f"ℹ️ Your price is broadly in line with the competitor (gap: {gap_pct:+.1f}%). "
            f"Competitive pricing risk is neutral."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature 7 — Promo Fatigue
# ─────────────────────────────────────────────────────────────────────────────

def render_promo_fatigue(r: dict) -> None:
    """
    Shows promo fatigue for the requested discount bucket.
    Warning banner when fatigue_flag fires. Expandable table for all buckets.
    """
    from engine.baseline_stats import bucket_discount

    fatigue_bucket = r.get("promo_fatigue_this_bucket", {})
    fatigue_flag   = r.get("promo_fatigue_flag", False)
    all_fatigue    = r.get("promo_fatigue_all_buckets", {})
    bucket_label   = bucket_discount(r.get("discount_pct", 0) / 100)

    if not fatigue_bucket:
        st.caption(f"🔁 Promo fatigue: **no data** for {bucket_label} discount bucket")
        return

    slope      = float(fatigue_bucket.get("slope", 0.0))
    obs        = int(fatigue_bucket.get("obs_count", 0))
    latest     = float(fatigue_bucket.get("latest_lift_pct", 0.0))
    earliest   = float(fatigue_bucket.get("earliest_lift_pct", 0.0))
    total_drop = latest - earliest

    if fatigue_flag:
        st.warning(
            f"🔁 **Promo Fatigue Detected** — lift at {bucket_label} off has been "
            f"declining by **{abs(slope):.1f}pp per run** across {obs} promotions. "
            f"First run: {earliest:.1f}% → Latest: {latest:.1f}% ({total_drop:+.1f}pp total).",
            icon=None,
        )
    else:
        color     = "green" if slope >= 0 else "orange"
        direction = "stable / improving" if slope >= 0 else f"mild decline ({slope:+.1f}pp/run)"
        st.markdown(
            f"**🔁 Promo Fatigue** &nbsp;·&nbsp; "
            f":{color}[{direction}] — {obs} historical run(s) at {bucket_label} off"
        )

    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("First-Run Lift",  f"{earliest:.1f}%",
               help="Lift% observed on the first historical promo in this bucket.")
    fc2.metric("Latest-Run Lift", f"{latest:.1f}%",
               delta=f"{total_drop:+.1f}pp total",
               delta_color="normal" if total_drop >= 0 else "inverse",
               help="Lift% on the most recent promo run.")
    fc3.metric("Trend per Run",   f"{slope:+.1f}pp",
               delta="fatigue" if fatigue_flag else "stable",
               delta_color="inverse" if fatigue_flag else "normal",
               help="OLS slope per successive promo. Negative = customers becoming desensitised.")

    if fatigue_flag:
        st.caption(
            f"ℹ️ Consider a deeper discount or a longer rest period before the next "
            f"{bucket_label} promo on this SKU to restore lift response."
        )

    if all_fatigue:
        with st.expander("📋 Fatigue across all discount buckets"):
            rows = [
                {
                    "Bucket":      b,
                    "Runs":        s.get("obs_count", 0),
                    "First Lift":  f"{s.get('earliest_lift_pct', 0):.1f}%",
                    "Latest Lift": f"{s.get('latest_lift_pct', 0):.1f}%",
                    "Trend/Run":   f"{s.get('slope', 0):+.1f}pp",
                    "Fatigue?":    "⚠️ Yes" if s.get("fatigue_flag") else "✓ No",
                }
                for b, s in all_fatigue.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
