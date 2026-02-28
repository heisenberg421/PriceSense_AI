"""
engine/verdict.py
-----------------
Step 7 of the engine pipeline.

Converts net profit delta + overall risk level into a clear, actionable verdict.

Decision tree:
    profitable + low risk    → RUN_IT
    profitable + medium risk → RUN_IT
    profitable + high risk   → RUN_WITH_CHANGES  (discount too deep or timing bad)
    not profitable + low risk  → RUN_WITH_CHANGES  (demand is there, math needs work)
    not profitable + medium risk → DONT_RUN
    not profitable + high risk   → DONT_RUN

Each verdict includes:
    verdict_key      : Internal identifier string
    label            : Human-readable verdict label
    icon             : Emoji for display
    streamlit_type   : Maps to st.success / st.warning / st.error
    summary          : One-sentence explanation shown in the badge
    suggested_change : Specific actionable tweak (None for RUN_IT)

No Streamlit imports. Pure Python.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Full verdict configurations. 
# streamlit_type is retained for downstream UI mapping (success/warning/error).
VERDICT_CONFIG: dict[str, dict] = {
    "RUN_IT": {
        "verdict_key":    "RUN_IT",
        "label":          "Run It",
        "icon":           "✅",
        "streamlit_type": "success",
        "summary":        "This promotion is projected to be profitable with manageable risk.",
    },
    "RUN_WITH_CHANGES": {
        "verdict_key":    "RUN_WITH_CHANGES",
        "label":          "Run with Changes",
        "icon":           "⚠️",
        "streamlit_type": "warning",
        "summary":        "The promotion has potential but requires tactical adjustments.",
    },
    "DONT_RUN": {
        "verdict_key":    "DONT_RUN",
        "label":          "Don't Run It",
        "icon":           "🚫",
        "streamlit_type": "error",
        "summary":        "This promotion is projected to destroy value or carries extreme risk.",
    },
}

# Actionable suggestions based on the specific failure point of the promo
_SUGGESTIONS: dict[str, str] = {
    "profitable_high_risk": (
        "Consider reducing the discount depth by 5% to ease margin compression, "
        "or shorten the promotional window to limit cannibalization exposure."
    ),
    "unprofitable_low_risk": (
        "Try a shallower discount (10–15%) or pair with a bundle to protect unit "
        "economics. The demand signal is there — the math just needs refinement."
    ),
    "unprofitable_other": (
        "Revisit discount depth and timing. A profitable path requires meaningfully "
        "reducing margin compression and secondary catalog cannibalization."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_verdict(
    net_profit_delta: float,
    overall_risk: str,
) -> dict:
    """
    Determine the final promotion verdict using a business decision tree.

    Decision Matrix:
    - Profitable + Low/Med Risk  -> RUN_IT
    - Profitable + High Risk     -> RUN_WITH_CHANGES (Safety First)
    - Unprofitable + Low Risk    -> RUN_WITH_CHANGES (Fix the Math)
    - Unprofitable + Med/High    -> DONT_RUN (Value Destruction)
    """

    profitable = net_profit_delta > 0
    suggested_change: str | None = None

    # Logic Path 1: Green Light
    if profitable and overall_risk in ("low", "medium"):
        key = "RUN_IT"

    # Logic Path 2: Profitable but Risky (e.g., deep discount on Nuts)
    elif profitable and overall_risk == "high":
        key = "RUN_WITH_CHANGES"
        suggested_change = _SUGGESTIONS["profitable_high_risk"]

    # Logic Path 3: Not profitable but safe (e.g., shallow discount, no lift)
    elif not profitable and overall_risk == "low":
        key = "RUN_WITH_CHANGES"
        suggested_change = _SUGGESTIONS["unprofitable_low_risk"]

    # Logic Path 4: Red Light (Unprofitable and risky)
    else:
        key = "DONT_RUN"
        suggested_change = _SUGGESTIONS["unprofitable_other"]

    

    return {
        **VERDICT_CONFIG[key],
        "suggested_change": suggested_change,
    }