"""
engine/data_analyzer.py
-----------------------
Thin orchestrator that wires together the modular data pipeline.

Sub-modules
-----------
    engine/baseline_stats.py  — per-SKU baseline, elasticity, promo-obs table
    engine/correlation.py     — cross-SKU Pearson correlation matrix
    engine/relationship.py    — hybrid TF-IDF + sequence + Jaccard classifier

Public API 
----------
    analyze_historical_data(df)  -> DataSummary
    compute_data_driven_lift(summary, sku_id, discount_pct) -> dict
    compute_data_driven_cannibalization(summary, promoted_sku_id,
                                        catalog_skus, incremental_units) -> dict

DataSummary is defined here and carries outputs from all sub-modules.
SKUSummary is re-exported from baseline_stats for downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from engine.baseline_stats import (
    SKUSummary, build_all_sku_stats, bucket_discount,
    SNR_HIGH_THRESHOLD, SNR_MEDIUM_THRESHOLD,
    compute_seasonality_index, compute_post_promo_dip, compute_promo_fatigue,
)
from engine.correlation import (
    build_correlation_matrix,
    get_correlation,
    cannibalization_bleed_rate,
)
from engine.relationship import classify_relationships, relationship_score_matrix


# ─────────────────────────────────────────────────────────────────────────────
# DataSummary — top-level container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataSummary:
    skus:        dict[str, SKUSummary]  # keyed by sku_id
    corr_matrix: pd.DataFrame           # Pearson correlation (sku × sku)
    sim_matrix:  pd.DataFrame           # blended name similarity (sku × sku)
    categories:  dict[str, list[str]]   # category label → [sku_ids]
    raw_df:      pd.DataFrame           # pivoted wide-form (week × sku = units)
    post_promo_dip:  dict[str, dict]    # sku_id → dip stats (Feature 5)
    promo_fatigue:   dict[str, dict]    # sku_id → fatigue stats per bucket (Feature 7)
    raw_hist_df: pd.DataFrame           # original long-form df (for seasonality lookup)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Analyse historical data
# ─────────────────────────────────────────────────────────────────────────────

def analyze_historical_data(df: pd.DataFrame) -> DataSummary:
    """
    Parse historical sales and compute signals for lift and cannibalization.
    """
    required = {"sku", "week", "price", "units_sold"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Historical data is missing required columns: {missing}")

    # Coerce types and handle optional columns
    df = df.copy()
    df["week"]       = pd.to_numeric(df["week"],       errors="coerce")
    df["price"]      = pd.to_numeric(df["price"],      errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df["promo_flag"] = pd.to_numeric(df.get("promo_flag", 0), errors="coerce").fillna(0).astype(int)
    df["discount"]   = pd.to_numeric(df.get("discount",   0), errors="coerce").fillna(0.0)
    df["cost"]       = pd.to_numeric(df.get("cost",       0), errors="coerce").fillna(0.0)
    
    # Ensure name and category strings are handled
    df["name"]       = df.get("name", df["sku"].astype(str)).fillna(df["sku"].astype(str))
    df["category"]   = df.get("category", "General").fillna("General")
    
    df.dropna(subset=["week", "price", "units_sold"], inplace=True)

    # 1. Per-SKU baseline stats + elasticity
    skus = build_all_sku_stats(df)

    # 2. Cross-SKU correlation matrix
    corr_matrix = build_correlation_matrix(df)

    # 3. Name-similarity matrix using REAL NAMES and CATEGORIES
    # We build a temporary catalog from the SKUSummary objects
    catalog_list = [
        {"id": sid, "name": s.name, "category": s.category} 
        for sid, s in skus.items()
    ]
    sim_matrix = relationship_score_matrix(catalog_list)

    # 4. Category membership map
    categories: dict[str, list[str]] = {}
    for sid, s in skus.items():
        categories.setdefault(s.category, [])
        if sid not in categories[s.category]:
            categories[s.category].append(sid)

    # 5. Pivoted raw data for internal diagnostic use
    raw_df = df.pivot_table(index="week", columns="sku", values="units_sold", aggfunc="mean")

    # 6. Feature 5: Post-promo dip per SKU
    post_promo_dip: dict[str, dict] = {}
    for sku_id, grp in df.groupby("sku"):
        post_promo_dip[str(sku_id)] = compute_post_promo_dip(grp.reset_index(drop=True))

    # 7. Feature 7: Promo fatigue per SKU
    promo_fatigue: dict[str, dict] = {}
    for sku_id, grp in df.groupby("sku"):
        promo_fatigue[str(sku_id)] = compute_promo_fatigue(grp.reset_index(drop=True))

    return DataSummary(
        skus         = skus,
        corr_matrix  = corr_matrix,
        sim_matrix   = sim_matrix,
        categories   = categories,
        raw_df       = raw_df,
        post_promo_dip = post_promo_dip,
        promo_fatigue  = promo_fatigue,
        raw_hist_df    = df,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Data-driven lift
# ─────────────────────────────────────────────────────────────────────────────

def compute_data_driven_lift(
    summary:      DataSummary,
    sku_id:       str,
    discount_pct: float,
    target_week:  int | None = None,
) -> dict:
    """
    Project unit lift using observed history or elasticity.

    Lift source priority
    --------------------
    1. Observed promos in matching discount bucket (≥3 obs)
       → avg_lift_pct is RECENCY-WEIGHTED (exponential decay, λ=0.02)
    2. OLS price elasticity model
    3. Fallback (discount × 1.2)

    Feature 4 — Seasonality
    -----------------------
    After selecting the lift%, the baseline is scaled by the seasonality index
    for the target_week (week % 52 position). This adjusts the incremental unit
    projection to reflect whether the promo runs in a peak or trough week.

    Confidence downgrade logic
    --------------------------
    After picking the base confidence, apply SNR guard:
        SNR = expected_lift_units / baseline_std
        SNR < 1.0                  → force LOW
        1.0 ≤ SNR < 2.0 AND high   → downgrade to MEDIUM

    New keys returned
    -----------------
    seasonality_index      : float  — multiplier applied to baseline (1.0 = neutral)
    seasonality_adjusted_baseline : float — baseline after seasonal scaling
    """
    sku = summary.skus.get(str(sku_id))
    if sku is None:
        return {
            "lift_pct": 0.0, "incremental_units": 0,
            "baseline_units": 100, "baseline_units_raw": 100,
            "trend_slope": 0.0, "trend_r2": 0.0,
            "recency_weighted_lift": None, "raw_avg_lift": None,
            "recency_effect": 0.0, "lift_snr": 0.0,
            "confidence": "low", "confidence_before_snr": "low", "method": "fallback",
            "seasonality_index": 1.0, "seasonality_adjusted_baseline": 100.0,
        }

    baseline = sku.baseline_units
    bucket   = bucket_discount(discount_pct / 100)
    obs_row  = (
        sku.promo_obs[sku.promo_obs["discount_bucket"] == bucket]
        if not sku.promo_obs.empty else pd.DataFrame()
    )

    recency_weighted_lift: float | None = None
    raw_avg_lift:          float | None = None
    recency_effect:        float        = 0.0

    if not obs_row.empty and int(obs_row.iloc[0]["obs_count"]) >= 3:
        lift_pct              = float(obs_row.iloc[0]["avg_lift_pct"])
        raw_avg_lift          = float(obs_row.iloc[0].get("raw_lift_pct", lift_pct))
        recency_effect        = float(obs_row.iloc[0].get("recency_effect", 0.0))
        recency_weighted_lift = lift_pct
        method                = "observed"
        confidence            = "high" if int(obs_row.iloc[0]["obs_count"]) >= 6 else "medium"
    elif sku.elasticity and sku.elasticity > 0:
        lift_pct   = sku.elasticity * discount_pct
        method     = "elasticity_model"
        confidence = "medium"
    else:
        lift_pct   = discount_pct * 1.2
        method     = "fallback"
        confidence = "low"

    lift_pct = min(lift_pct, 200.0)

    # ── Feature 4: Apply seasonality index ────────────────────────────────────
    sku_hist = summary.raw_hist_df[summary.raw_hist_df["sku"] == str(sku_id)]
    seasonality_index = compute_seasonality_index(sku_hist.reset_index(drop=True), target_week)
    seasonality_adjusted_baseline = round(baseline * seasonality_index, 1)

    incremental_units = max(0, round(seasonality_adjusted_baseline * lift_pct / 100))

    # ── SNR confidence downgrade ───────────────────────────────────────────────
    confidence_before_snr = confidence
    if sku.baseline_std > 0:
        snr = incremental_units / sku.baseline_std
        if snr < SNR_MEDIUM_THRESHOLD:
            confidence = "low"
        elif snr < SNR_HIGH_THRESHOLD and confidence == "high":
            confidence = "medium"
    else:
        snr = 99.0   # zero std = perfectly stable demand

    return {
        "lift_pct":            round(lift_pct, 1),
        "incremental_units":   incremental_units,
        "baseline_units":      round(seasonality_adjusted_baseline, 1),
        "baseline_units_raw":  round(sku.baseline_units_raw, 1),
        "trend_slope":         sku.trend_slope,
        "trend_r2":            sku.trend_r2,
        "recency_weighted_lift": round(recency_weighted_lift, 1) if recency_weighted_lift is not None else None,
        "raw_avg_lift":          round(raw_avg_lift, 1)          if raw_avg_lift          is not None else None,
        "recency_effect":        round(recency_effect, 1),
        "lift_snr":              round(snr, 2),
        "confidence":            confidence,
        "confidence_before_snr": confidence_before_snr,
        "method":                method,
        "seasonality_index":     round(seasonality_index, 3),
        "seasonality_adjusted_baseline": round(seasonality_adjusted_baseline, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Data-driven cannibalization
# ─────────────────────────────────────────────────────────────────────────────

def compute_data_driven_cannibalization(
    summary:           DataSummary,
    promoted_sku_id:   str,
    catalog_skus:      list[dict],
    incremental_units: int,
) -> dict:
    """
    Estimate cannibalization using correlation + classified relationships.
    """
    p_id = str(promoted_sku_id)
    if not summary.skus.get(p_id) or incremental_units == 0:
        return {
            "total_cannibalized_units":   0,
            "total_cannibalized_revenue": 0.0,
            "cannibalized_pct":           0.0,
            "net_incremental_units":      incremental_units,
            "sku_breakdown":              [],
        }

    total_revenue_impact = 0.0
    sku_breakdown: list[dict] = []

    for sku in catalog_skus:
        other_id = str(sku.get("id", sku.get("sku", "")))
        rel      = sku.get("relationship", "unrelated")

        if rel == "promoted" or other_id == p_id:
            continue

        # Get correlation from the Pearson matrix
        corr_val   = get_correlation(summary.corr_matrix, p_id, other_id)
        
        # Calculate bleed based on updated correlation logic (engine/correlation.py)
        bleed_rate = cannibalization_bleed_rate(rel, corr_val)

        # if bleed_rate == 0.0:
        #     continue

        # 
        lost_units     = round(incremental_units * bleed_rate)
        price          = float(sku.get("price", 0.0))
        margin         = float(sku.get("margin", 40)) / 100.0
        
        # Profit-based impact for the cannibalization delta
        revenue_impact = lost_units * price * margin
        total_revenue_impact += revenue_impact

        sku_breakdown.append({
            "id":             other_id,
            "name":           str(sku.get("name", other_id)),
            "relationship":   rel,
            "lost_units":     abs(lost_units),
            "direction":      "loss" if bleed_rate >= 0 else "gain",
            "revenue_impact": round(revenue_impact, 2),
        })

    total_cannibalized_units = sum(
        s["lost_units"] for s in sku_breakdown if s["direction"] == "loss"
    )
    cannibalized_pct = (
        round(total_cannibalized_units / incremental_units * 100.0, 1)
        if incremental_units > 0 else 0.0
    )

    return {
        "total_cannibalized_units":   total_cannibalized_units,
        "total_cannibalized_revenue": round(total_revenue_impact, 2),
        "cannibalized_pct":           cannibalized_pct,
        "net_incremental_units":      max(0, incremental_units - total_cannibalized_units),
        "sku_breakdown":              sku_breakdown,
    }