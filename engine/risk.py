"""
engine/risk.py
--------------
Step 6 of the engine pipeline.

Produces a composite risk score from three independent signals:

    Signal 1 — Margin Compression Risk
        How much of the promoted SKU's own margin does the discount consume?
        Metric: discount_pct / margin_pct
            < 0.40  → low
            0.40–0.75 → medium
            > 0.75  → high

    Signal 2 — Cannibalization Risk
        What fraction of the gross lift bleeds to other catalog SKUs?
        Metric: cannibalized_pct
            < 15%   → low
            15–35%  → medium
            > 35%   → high

    Signal 3 — Timing / Context Risk
        Keyword scan of the free-text context field.
        Risk-raisers: competitor, excess inventory, overstocked, slow season, etc.
        Risk-lowerers: high demand, event, festival, holiday, seasonal peak, etc.
        Score:
            net score > 0  → high
            net score == 0 → medium
            net score < 0  → low

    Signal 4 — Competitor Price Gap  [NEW — Feature 6]
        Uses the competitor_price column from the historical CSV (if present).
        Gap = (your_full_price − competitor_price) / your_full_price × 100
        Positive gap → you are more expensive than competitor → higher timing risk
        Negative gap → you are already cheaper               → lower timing risk
        Thresholds:
            gap < −5%   → low   (already undercutting)
            gap > +10%  → high  (significantly overpriced vs. competitor)
            otherwise   → medium (neutral)
        If no competitor_price data is available, returns None (signal is skipped).

Composite:
    Each signal maps to a weight: low=1, medium=2, high=3
    Sum of all three weights (range 3–9):
        3–4 → overall low
        5–6 → overall medium
        7–9 → overall high

No Streamlit imports. Pure Python.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Configuration & Thresholds
# ─────────────────────────────────────────────────────────────────────────────

RISK_WEIGHTS: dict[str, int] = {"low": 1, "medium": 2, "high": 3}

# Per-category thresholds for Signal 1 (Margin Compression)
# Format: {category: (low_threshold, medium_threshold)}
CATEGORY_MARGIN_RATIO_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Nuts":      (0.35, 0.65),
    "Beverage":  (0.40, 0.72),
    "Lotion":    (0.42, 0.70),
    "Yogurt":    (0.45, 0.80),
    "Vitamins":  (0.50, 0.85),
    "Dairy":     (0.45, 0.80),
    "General":   (0.40, 0.75), # Fallback
}

# Per-category thresholds for Signal 2 (Cannibalization Bleed %)
CATEGORY_CANNIBALIZATION_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Nuts":      (12.0, 30.0),
    "Beverage":  (16.0, 36.0),
    "Lotion":    (15.0, 34.0),
    "Yogurt":    (18.0, 38.0),
    "Vitamins":  (14.0, 32.0),
    "Dairy":     (18.0, 38.0),
    "General":   (15.0, 35.0), # Fallback
}

# Signal 3 Keywords
CONTEXT_RISK_RAISERS = [
    "competitor", "excess inventory", "overstocked", "slow season", 
    "low demand", "clearance needed", "margin squeeze"
]

CONTEXT_RISK_LOWERERS = [
    "high demand", "event", "festival", "holiday", "seasonal peak",
    "back to school", "black friday", "cyber monday", "sell-through"
]

# ─────────────────────────────────────────────────────────────────────────────
# Internal Signal Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _margin_compression_risk(discount_pct: float, margin_pct: float, category_type: str) -> str:
    """Signal 1: Logic to detect if a discount is eating too much margin."""
    if margin_pct <= 0: return "high"
    
    low_thr, med_thr = CATEGORY_MARGIN_RATIO_THRESHOLDS.get(category_type, (0.40, 0.75))
    ratio = discount_pct / margin_pct
    
    if ratio < low_thr:  return "low"
    if ratio < med_thr:  return "medium"
    return "high"

def _cannibalization_risk(cannibalized_pct: float, category_type: str) -> str:
    """Signal 2: Logic to detect if the promotion is cannibalizing the catalog."""
    low_thr, med_thr = CATEGORY_CANNIBALIZATION_THRESHOLDS.get(category_type, (15.0, 35.0))
    
    if cannibalized_pct < low_thr: return "low"
    if cannibalized_pct < med_thr: return "medium"
    return "high"

def _parse_context_risk(context_text: str) -> str:
    """Signal 3: Sentiment/Keyword scan of the free-text context."""
    text = context_text.lower().strip()
    if not text or text == "none": return "medium"
    
    score = 0
    for kw in CONTEXT_RISK_RAISERS:
        if kw in text: score += 1
    for kw in CONTEXT_RISK_LOWERERS:
        if kw in text: score -= 1

    if score > 0:  return "high"
    if score < 0:  return "low"
    return "medium"

def _competitor_price_risk(
    full_price: float,
    competitor_price: float | None,
) -> str | None:
    """
    Signal 4: Assess timing risk based on price gap versus competitor.

    Gap = (your_price − competitor_price) / your_price × 100
      Positive → you are more expensive (higher risk to run without promo)
      Negative → you are cheaper (low risk, promo may be unnecessary)

    Returns None if competitor_price is unavailable.
    """
    if competitor_price is None or competitor_price <= 0 or full_price <= 0:
        return None

    gap_pct = (full_price - competitor_price) / full_price * 100

    if gap_pct > 10.0:
        return "high"    # meaningfully more expensive than competitor
    if gap_pct < -5.0:
        return "low"     # already undercutting competitor
    return "medium"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk(
    discount_pct:      float,
    margin_pct:        float,
    cannibalized_pct:  float,
    context_text:      str,
    category_type:     str,
    full_price:        float = 0.0,
    competitor_price:  float | None = None,
) -> dict:
    """
    Produces a multi-signal risk assessment for the proposed promotion.

    competitor_price is optional. If provided, a 4th signal (competitor gap)
    is incorporated into the composite score.
    """
    m_risk = _margin_compression_risk(discount_pct, margin_pct, category_type)
    c_risk = _cannibalization_risk(cannibalized_pct, category_type)
    t_risk = _parse_context_risk(context_text)
    p_risk = _competitor_price_risk(full_price, competitor_price)

    # Composite logic: Weighted sum of signals
    composite_score = RISK_WEIGHTS[m_risk] + RISK_WEIGHTS[c_risk] + RISK_WEIGHTS[t_risk]

    # Add competitor price signal if available (same weighting as other signals)
    if p_risk is not None:
        composite_score += RISK_WEIGHTS[p_risk]
        max_score = 12   # 4 signals × max 3 each
        # Re-scale thresholds to 4-signal range (4–12)
        if composite_score <= 6:    overall = "low"
        elif composite_score <= 9:  overall = "medium"
        else:                       overall = "high"
    else:
        # Original 3-signal scoring (3–9)
        if composite_score <= 4:    overall = "low"
        elif composite_score <= 6:  overall = "medium"
        else:                       overall = "high"

    return {
        "margin_risk":           m_risk,
        "cannibalization_risk":  c_risk,
        "timing_risk":           t_risk,
        "competitor_price_risk": p_risk,
        "overall_risk":          overall,
        "risk_score":            composite_score,
    }