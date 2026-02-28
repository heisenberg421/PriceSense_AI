"""
engine/correlation.py
---------------------
Builds and queries the cross-SKU weekly units correlation matrix.

The correlation matrix captures how SKU sales move together over time.
High positive correlation during the same promo week suggests one SKU
cannibalises another when promoted. Low or negative correlation implies
independence or complementarity.

Public API
----------
    build_correlation_matrix(df) -> pd.DataFrame
        Pivot weekly units into a wide matrix and compute Pearson correlations.

    get_correlation(matrix, sku_a, sku_b) -> float
        Safe accessor — returns 0.0 if either SKU is missing from the matrix.

    cannibalization_bleed_rate(rel, corr_val) -> float
        Maps (relationship_label, correlation_value) → bleed_rate fraction.
        Used by the cannibalization engine to size the bleed for each SKU pair.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Base bleed rates by relationship type
# ─────────────────────────────────────────────────────────────────────────────
# 
# These fractions represent how much of the promoted SKU's incremental lift 
# is "stolen" from another SKU.
#
_BASE_BLEED: dict[str, float] = {
    "variant":    0.18,   # Same product, different size: high switching probability
    "substitute": 0.08,   # Different product, same category: moderate switching
    "complement": -0.05,  # Halo effect: promoting SKU A actually helps SKU B
    "unrelated":  0.00,   # No structural relationship
    "promoted":   0.00,   # Focal SKU
}

# Ceiling bleed rates: The maximum amount correlation can inflate the bleed
_BLEED_CEILING: dict[str, float] = {
    "variant":    0.40,   # If correlation is +1.0, up to 40% of lift can be cannibalization
    "substitute": 0.20,   # Competitive substitutes cap at 20%
    "complement": 0.00,   # Complements can only have negative bleed (gains)
    "unrelated":  0.05,   # Residual bleed for high-correlation outliers
}

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot weekly units into a wide (week × sku) matrix and compute Pearson correlations.
    """
    pivot = df.pivot_table(
        index="week",
        columns="sku",
        values="units_sold",
        aggfunc="mean",
    )
    # Ensure we only correlate numeric data and handle missing weeks
    return pivot.corr(method="pearson", min_periods=3)


def get_correlation(matrix: pd.DataFrame, sku_a: str, sku_b: str) -> float:
    """
    Safely retrieve the correlation between two SKUs.
    Returns 0.0 if either SKU is missing or if the value is NaN.
    """
    try:
        val = matrix.loc[str(sku_a), str(sku_b)]
        return float(val) if not pd.isna(val) else 0.0
    except (KeyError, ValueError):
        return 0.0


def cannibalization_bleed_rate(rel: str, corr_val: float) -> float:
    """
    Maps (relationship_label, correlation_value) → bleed_rate fraction.
    
    Logic:
        1. Start with the _BASE_BLEED for the relationship.
        2. Adjust by the Pearson correlation (observed behavior).
        3. Clamp to _BLEED_CEILING to prevent unrealistic data noise.
    """
    # 
    
    # 1. Unrelated Guard
    if rel in ("promoted", "unrelated"):
        if rel == "unrelated" and corr_val > 0.5:
            # Only unrelated items with very high correlation bleed
            return min(_BLEED_CEILING["unrelated"], corr_val * 0.05)
        return 0.0

    base    = _BASE_BLEED.get(rel, 0.0)
    ceiling = _BLEED_CEILING.get(rel, 0.10)

    # 2. Complement (Halo Effect) Logic
    if rel == "complement":
        # Positive correlation with a complement means stronger Halo (more negative bleed)
        # We use a 0.5 multiplier to keep the halo effect conservative
        modifier = 1.0 + (max(0.0, corr_val) * 0.5)
        rate = base * modifier
        return max(-0.12, min(0.0, rate))

    # 3. Variant & Substitute (Cannibalization) Logic
    # High positive correlation confirms that these items move in sync
    # If they both spike during a promo, it's a true substitute/variant switching scenario
    corr_modifier = 1.0 + (corr_val * 0.4) 
    rate = base * max(0.6, corr_modifier)
    
    return min(ceiling, round(rate, 4))