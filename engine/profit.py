"""
engine/profit.py
----------------
Step 5 of the engine pipeline.

Computes net incremental profit from running the promotion vs. not running it,
accounting for:
    1. Discount-induced margin compression on the promoted SKU
    2. Revenue uplift from increased volume
    3. Cannibalization losses from catalog bleed (passed in from Step 3)

Formula:
    discounted_price     = full_price × (1 - discount_pct / 100)
    baseline_revenue     = baseline_units × full_price
    promo_revenue        = (baseline_units + incremental_units) × discounted_price
    baseline_profit      = baseline_revenue × (margin_pct / 100)
    effective_margin     = max(margin_pct/100 - discount_pct/100, 0)
    promo_profit         = promo_revenue × effective_margin
    revenue_uplift       = promo_revenue - baseline_revenue
    post_promo_dip_loss  = dip_units × full_price × effective_margin   [Feature 5]
    net_profit_delta     = promo_profit - baseline_profit - cannibalized_revenue_loss
                           - post_promo_dip_loss

Key insight on effective_margin:
    A 25% discount on a 52% margin product leaves 27% effective margin.
    A 25% discount on a 20% margin product leaves 0% effective margin (floored at 0).
    This is why deep discounts on low-margin products almost always fail.

No Streamlit imports. Pure Python.
"""

def compute_profit(
    baseline_units: float,
    incremental_units: int,
    full_price: float,
    discount_pct: float,
    margin_pct: float,
    cannibalized_revenue_loss: float,
    post_promo_dip_units: float = 0.0,
) -> dict:
    """
    Compute net profit delta from running the promotion.

    Args:
        baseline_units            : Expected units at full price.
        incremental_units         : Projected gross unit lift.
        full_price                : Standard retail price.
        discount_pct              : Discount applied (e.g., 20.0 for 20% off).
        margin_pct                : SKU's gross margin (e.g., 50.0 for 50%).
        cannibalized_revenue_loss : Profit lost to other SKUs (from cannibalization engine).
        post_promo_dip_units      : Average units lost in weeks after promo [Feature 5].

    Returns:
        dict: Financial metrics for the proposed promotion.
    """
    # 

    # Calculate promo price
    discount_fraction = discount_pct / 100.0
    discounted_price = full_price * (1.0 - discount_fraction)
    
    # Revenue Logic
    baseline_revenue = baseline_units * full_price
    promo_revenue    = (baseline_units + incremental_units) * discounted_price
    revenue_uplift   = promo_revenue - baseline_revenue

    # Margin Logic
    margin_fraction = margin_pct / 100.0
    # effective_margin represents the "cents on the dollar" kept after discount
    effective_margin = max(margin_fraction - discount_fraction, 0.0)

    # Profit Logic
    # Baseline: What we would have made anyway
    baseline_profit = baseline_revenue * margin_fraction
    
    # Promo: What we make on the total volume at the new lower margin
    promo_profit = promo_revenue * effective_margin

    # Net Profit Delta: This is the critical "Verdict" number
    # It accounts for the gain/loss on the product itself MINUS cannibalization bleed
    # MINUS post-promo demand hangover [Feature 5]
    post_promo_dip_loss = post_promo_dip_units * full_price * effective_margin
    net_profit_delta = promo_profit - baseline_profit - cannibalized_revenue_loss - post_promo_dip_loss

    return {
        "discounted_price":      round(discounted_price, 2),
        "baseline_revenue":      round(baseline_revenue, 2),
        "promo_revenue":         round(promo_revenue, 2),
        "baseline_profit":       round(baseline_profit, 2),
        "promo_profit":          round(promo_profit, 2),
        "revenue_uplift":        round(revenue_uplift, 2),
        "post_promo_dip_loss":   round(post_promo_dip_loss, 2),
        "net_profit_delta":      round(net_profit_delta, 2),
    }