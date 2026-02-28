"""
engine/baseline_stats.py
------------------------
Computes per-SKU baseline statistics from historical sales data.

Features
--------
    1. Trend-adjusted baseline
       OLS trend projected to next week, blended with flat mean via R².

    2. Recency-weighted promo observations  [NEW]
       Each promo observation is weighted by exponential decay so recent
       promos count more than old ones.
       λ = DECAY_LAMBDA (default 0.02) → half-life ≈ 35 weeks.
       Stored in promo_obs as column `weighted_lift_pct` alongside `avg_lift_pct`.
       compute_data_driven_lift() uses weighted_lift_pct when available.

    3. baseline_std → SNR guard  [NEW]
       Signal-to-noise ratio = expected_lift_units / baseline_std.
       Stored as `lift_snr` in SKUSummary and returned from lift.
       The caller (data_analyzer) uses it to downgrade confidence.

SKUSummary 
-----------------------------------------
    baseline_units_raw : float   
    trend_slope        : float   
    trend_r2           : float   
    lift_snr           : float   

promo_obs 
---------------------
    weighted_lift_pct  : float   — exponentially-decayed weighted average lift
    raw_lift_pct       : float   — original simple average (kept for display)
    
Public API
----------
    compute_sku_stats(sku_id, df_sku) -> SKUSummary
    build_all_sku_stats(df) -> dict[str, SKUSummary]
    bucket_discount(d) -> str
    compute_seasonality_index(df_sku, target_week) -> float
    DECAY_LAMBDA : float   (importable constant for UI display)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DECAY_LAMBDA: float = 0.02
# Exponential decay rate for promo recency weighting.
# half-life = ln(2) / λ ≈ 35 weeks (~8 months).
# Promos from 2 years ago receive ~12% of the weight of a current promo.

SNR_HIGH_THRESHOLD:   float = 2.0   # SNR ≥ 2.0 → no downgrade
SNR_MEDIUM_THRESHOLD: float = 1.0   # SNR ≥ 1.0 < 2.0 → cap at medium
# SNR < 1.0 → downgrade to low (lift is within one std of baseline noise)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SKUSummary:
    sku_id:             str
    name:               str
    category:           str
    full_price:         float   # mean price in non-promo weeks
    unit_cost:          float   # mean cost across all weeks
    margin_pct:         float   # (full_price - unit_cost) / full_price × 100
    baseline_units:     float   # trend-adjusted weekly baseline (PRIMARY)
    baseline_units_raw: float   # flat all-time mean (for audit / display)
    baseline_std:       float   # std of non-promo weekly units
    trend_slope:        float   # OLS slope: units per week
    trend_r2:           float   # R² of trend fit [0, 1]
    lift_snr:           float   # signal-to-noise at reference 25% discount
    promo_obs:          pd.DataFrame  # per-bucket lift table (see column docs above)
    elasticity:         float   # abs(Δ%units / Δ%price) from OLS


# ─────────────────────────────────────────────────────────────────────────────
# Discount bucketing
# ─────────────────────────────────────────────────────────────────────────────

def bucket_discount(d: float) -> str:
    """Map a discount fraction (0–1) to a human-readable range label."""
    pct = d * 100
    if pct < 5:   return "0–5%"
    if pct < 12:  return "5–12%"
    if pct < 18:  return "12–18%"
    if pct < 25:  return "18–25%"
    if pct < 33:  return "25–33%"
    return "33%+"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_trend(base_rows: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Fit linear OLS trend through non-promo weekly units.

    Returns (trend_baseline, mean_units, slope, r2).
    trend_baseline is R²-blended: high R² trusts the projection,
    low R² stays near the flat mean.
    """
    if base_rows.empty:
        return 0.0, 0.0, 0.0, 0.0

    x = base_rows["week"].values.astype(float)
    y = base_rows["units_sold"].values.astype(float)
    mean_units = float(y.mean())

    if len(x) < 8 or mean_units == 0:
        return mean_units, mean_units, 0.0, 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coef = np.polyfit(x, y, 1)

    slope, intercept = float(coef[0]), float(coef[1])

    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - mean_units) ** 2))
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    next_week = float(x.max()) + 1.0
    projected = slope * next_week + intercept
    projected = max(1.0, min(projected, 3.0 * mean_units))

    trend_baseline = r2 * projected + (1.0 - r2) * mean_units
    return trend_baseline, mean_units, round(slope, 4), round(r2, 4)


def _build_promo_obs_table(
    df_sku: pd.DataFrame,
    baseline_units: float,
    decay_lambda: float = DECAY_LAMBDA,
) -> pd.DataFrame:
    """
    Build a discount-bucket summary table with BOTH simple and
    recency-weighted average lift.

    Columns returned
    ----------------
    discount_bucket   : str    — bucket label
    avg_lift_pct      : float  — recency-weighted lift (PRIMARY — used by lift engine)
    raw_lift_pct      : float  — simple unweighted average (for display / audit)
    obs_count         : int    — number of promo observations in this bucket
    recency_effect    : float  — avg_lift_pct − raw_lift_pct (shows direction of correction)

    Design
    ------
    Weight for each promo observation:
        w_i = exp(−λ × (max_week − week_i))
    Weighted average:
        weighted_lift = Σ(lift_i × w_i) / Σ(w_i)

    `avg_lift_pct` == weighted average so all existing code reading
    `avg_lift_pct` automatically picks up the recency-corrected value.
    `raw_lift_pct` is kept separately for the UI to show the delta.
    """
    promo = df_sku[df_sku["promo_flag"] == 1].copy()
    if promo.empty or baseline_units == 0:
        return pd.DataFrame(
            columns=["discount_bucket", "avg_lift_pct", "raw_lift_pct",
                     "obs_count", "recency_effect"]
        )

    promo["lift_pct"]        = (promo["units_sold"] - baseline_units) / baseline_units * 100
    promo["discount_bucket"] = promo["discount"].apply(bucket_discount)

    max_week = float(promo["week"].max())

    def _weighted_stats(grp: pd.DataFrame) -> pd.Series:
        weights     = np.exp(-decay_lambda * (max_week - grp["week"].values))
        w_sum       = weights.sum()
        w_lift      = float((grp["lift_pct"].values * weights).sum() / w_sum) if w_sum > 0 else float(grp["lift_pct"].mean())
        simple_lift = float(grp["lift_pct"].mean())
        return pd.Series({
            "avg_lift_pct":  round(w_lift, 4),        # weighted (PRIMARY)
            "raw_lift_pct":  round(simple_lift, 4),   # simple average (audit)
            "obs_count":     len(grp),
            "recency_effect": round(w_lift - simple_lift, 4),
        })

    return (
        promo.groupby("discount_bucket", group_keys=False)
        .apply(_weighted_stats)
        .reset_index()
    )


def _compute_elasticity(df_sku: pd.DataFrame) -> float:
    """OLS price elasticity. Returns abs(Δ%units / Δ%price). Falls back to 1.5."""
    promo      = df_sku[df_sku["promo_flag"] == 1].copy()
    base_rows  = df_sku[df_sku["promo_flag"] == 0]
    base_units = base_rows["units_sold"].mean() if not base_rows.empty else 0.0
    base_price = base_rows["price"].mean()      if not base_rows.empty else 0.0

    if len(promo) < 4 or base_units == 0 or base_price == 0:
        return 1.5

    promo = promo.copy()
    promo["pct_price_change"] = (promo["price"] - base_price) / base_price * 100
    promo["pct_unit_change"]  = (promo["units_sold"] - base_units) / base_units * 100

    mask = (
        (np.abs(promo["pct_price_change"]) < 50) &
        (np.abs(promo["pct_unit_change"])  < 200)
    )
    promo = promo[mask]
    if len(promo) < 3:
        return 1.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coef = np.polyfit(promo["pct_price_change"], promo["pct_unit_change"], 1)

    return round(max(0.5, min(abs(float(coef[0])), 5.0)), 3)


def _compute_lift_snr(
    baseline_units: float,
    baseline_std: float,
    elasticity: float,
    reference_discount: float = 25.0,
) -> float:
    """
    Compute signal-to-noise ratio for lift at a reference discount depth.

    SNR = expected_lift_units / baseline_std
    Higher SNR → lift is clearly above background demand noise.
    Lower SNR → lift is hard to distinguish from normal weekly variance.

    Returns 0.0 if std is zero (perfectly stable demand — effectively infinite SNR,
    we cap display at 99 in UI).
    """
    if baseline_std <= 0:
        return 99.0   # stable demand, no noise concern
    expected_lift_units = baseline_units * (elasticity * reference_discount) / 100.0
    return round(expected_lift_units / baseline_std, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_sku_stats(sku_id: str, df_sku: pd.DataFrame) -> SKUSummary:
    """
    Compute all baseline statistics for a single SKU.

    Key outputs:
        baseline_units     → trend-adjusted (PRIMARY for all lift/profit calcs)
        baseline_units_raw → flat all-time mean (audit only)
        promo_obs          → per-bucket table; avg_lift_pct = recency-weighted
        lift_snr           → signal-to-noise at reference 25% discount
    """
    base_rows = df_sku[df_sku["promo_flag"] == 0]

    name     = str(df_sku["name"].iloc[0])     if "name"     in df_sku.columns else sku_id
    category = str(df_sku["category"].iloc[0]) if "category" in df_sku.columns else "General"

    full_price   = float(base_rows["price"].mean() if not base_rows.empty else df_sku["price"].max())
    unit_cost    = float(df_sku["cost"].mean())
    margin_pct   = ((full_price - unit_cost) / full_price * 100) if full_price > 0 else 0.0
    baseline_std = float(base_rows["units_sold"].std() if not base_rows.empty else 0.0)

    # 1. Trend-adjusted baseline
    trend_baseline, mean_units, slope, r2 = _compute_trend(base_rows.sort_values("week"))
    if trend_baseline == 0.0:
        mean_units     = float(df_sku["units_sold"].mean())
        trend_baseline = mean_units

    # 2. Elasticity
    elasticity = _compute_elasticity(df_sku)

    # 3. Recency-weighted promo obs (avg_lift_pct = weighted)
    promo_obs = _build_promo_obs_table(df_sku, trend_baseline)

    # 4. SNR at reference 25% discount
    lift_snr = _compute_lift_snr(trend_baseline, baseline_std, elasticity)

    return SKUSummary(
        sku_id             = sku_id,
        name               = name,
        category           = category,
        full_price         = round(full_price, 2),
        unit_cost          = round(unit_cost, 2),
        margin_pct         = round(margin_pct, 1),
        baseline_units     = round(trend_baseline, 1),
        baseline_units_raw = round(mean_units, 1),
        baseline_std       = round(baseline_std, 1),
        trend_slope        = slope,
        trend_r2           = r2,
        lift_snr           = lift_snr,
        promo_obs          = promo_obs,
        elasticity         = elasticity,
    )


def build_all_sku_stats(df: pd.DataFrame) -> dict[str, SKUSummary]:
    """Compute SKUSummary for every SKU in the historical DataFrame."""
    return {
        str(sku_id): compute_sku_stats(str(sku_id), grp.reset_index(drop=True))
        for sku_id, grp in df.groupby("sku")
    }

# ─────────────────────────────────────────────────────────────────────────────
# Feature 4: Seasonality Index
# ─────────────────────────────────────────────────────────────────────────────

def compute_seasonality_index(df_sku: pd.DataFrame, target_week: int | None = None) -> float:
    """
    Compute a seasonality multiplier for a given target week.

    Using non-promo weeks only, calculates the ratio of units at the target
    week-of-year position to the overall annual mean.

      seasonality_index > 1.0 → this week is typically above average (peak)
      seasonality_index < 1.0 → this week is typically below average (trough)
      seasonality_index = 1.0 → neutral / insufficient data

    If target_week is None, uses next week after the last observed week.
    Clamped to [0.5, 2.0].
    """
    base_rows = df_sku[df_sku["promo_flag"] == 0].copy()
    if base_rows.empty or len(base_rows) < 12:
        return 1.0

    annual_mean = float(base_rows["units_sold"].mean())
    if annual_mean == 0:
        return 1.0

    if target_week is None:
        target_week = int(base_rows["week"].max()) + 1

    target_position = int(target_week) % 52
    base_rows["week_position"] = base_rows["week"].astype(int) % 52

    # ±2-week window for robustness
    window = 2
    positions = [(target_position + offset) % 52 for offset in range(-window, window + 1)]
    nearby = base_rows[base_rows["week_position"].isin(positions)]

    if len(nearby) < 3:
        return 1.0

    position_mean = float(nearby["units_sold"].mean())
    index = round(position_mean / annual_mean, 3)
    return float(max(0.5, min(index, 2.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5: Post-Promo Dip Modelling
# ─────────────────────────────────────────────────────────────────────────────

def compute_post_promo_dip(df_sku: pd.DataFrame) -> dict:
    """
    Estimate the post-promotion demand hangover (pantry-loading effect).

    For each historical promo week, measure the unit deficit in weeks t+1
    and t+2 versus the non-promo baseline. Only counts deficits (negative
    deviations); above-baseline post-promo weeks are treated as 0.

    Returns
    -------
    dict:
        dip_units_week1  : float  — avg units BELOW baseline in week after promo
        dip_units_week2  : float  — avg units BELOW baseline 2 weeks after promo
        total_dip_units  : float  — combined average deficit across both weeks
        dip_obs_count    : int    — number of promo-event pairs measured
        dip_confidence   : str    — 'high' (≥5), 'medium' (3–4), 'low' (<3)
    """
    base_rows = df_sku[df_sku["promo_flag"] == 0]
    baseline  = float(base_rows["units_sold"].mean()) if not base_rows.empty else 0.0

    promo_weeks = sorted(df_sku[df_sku["promo_flag"] == 1]["week"].unique())

    dips_w1: list[float] = []
    dips_w2: list[float] = []

    for pw in promo_weeks:
        w1_row = df_sku[df_sku["week"] == pw + 1]
        w2_row = df_sku[df_sku["week"] == pw + 2]

        w1_promo = (not w1_row.empty and int(w1_row.iloc[0].get("promo_flag", 0)) == 1)
        if not w1_promo and not w1_row.empty:
            dips_w1.append(float(w1_row.iloc[0]["units_sold"]) - baseline)

        w2_promo = (not w2_row.empty and int(w2_row.iloc[0].get("promo_flag", 0)) == 1)
        if not w2_promo and not w2_row.empty:
            dips_w2.append(float(w2_row.iloc[0]["units_sold"]) - baseline)

    def _safe_mean(lst: list[float]) -> float:
        return round(float(sum(lst) / len(lst)), 2) if lst else 0.0

    dip_w1_loss = min(_safe_mean(dips_w1), 0.0)
    dip_w2_loss = min(_safe_mean(dips_w2), 0.0)
    n = min(len(dips_w1), len(dips_w2)) if (dips_w1 and dips_w2) else max(len(dips_w1), len(dips_w2))
    confidence  = "high" if n >= 5 else ("medium" if n >= 3 else "low")

    return {
        "dip_units_week1": abs(dip_w1_loss),
        "dip_units_week2": abs(dip_w2_loss),
        "total_dip_units": round(abs(dip_w1_loss) + abs(dip_w2_loss), 2),
        "dip_obs_count":   n,
        "dip_confidence":  confidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 7: Promo Fatigue Detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_promo_fatigue(df_sku: pd.DataFrame) -> dict:
    """
    Detect declining lift effectiveness from repeated promos at the same depth.

    For each discount bucket, fits an OLS trend across successive promo runs
    (x = promo index chronologically, y = lift%). A negative slope signals
    diminishing returns / promo fatigue.

    Returns
    -------
    dict keyed by discount_bucket:
        slope            : float — lift% change per successive promo (negative = fatigue)
        obs_count        : int
        latest_lift_pct  : float — most recent lift
        earliest_lift_pct: float — first observed lift
        fatigue_flag     : bool  — True if slope < −1.5 and obs_count ≥ 3
    """
    base_rows = df_sku[df_sku["promo_flag"] == 0]
    baseline  = float(base_rows["units_sold"].mean()) if not base_rows.empty else 0.0
    if baseline == 0:
        return {}

    promo = df_sku[df_sku["promo_flag"] == 1].copy()
    if promo.empty:
        return {}

    promo["lift_pct"]        = (promo["units_sold"] - baseline) / baseline * 100
    promo["discount_bucket"] = promo["discount"].apply(bucket_discount)

    result: dict = {}
    for bucket, grp in promo.sort_values("week").groupby("discount_bucket"):
        lifts = grp["lift_pct"].values.astype(float)
        n     = len(lifts)
        x     = np.arange(1, n + 1, dtype=float)

        if n >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coef  = np.polyfit(x, lifts, 1)
            slope = round(float(coef[0]), 2)
        else:
            slope = 0.0

        result[str(bucket)] = {
            "slope":              slope,
            "obs_count":          n,
            "latest_lift_pct":    round(float(lifts[-1]), 1),
            "earliest_lift_pct":  round(float(lifts[0]),  1),
            "fatigue_flag":       slope < -1.5 and n >= 3,
        }

    return result
