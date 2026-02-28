"""
engine/orchestrator.py
----------------------
Data-driven promotion analysis pipeline.

Pipeline
--------
    Step 1 → data_analyzer.analyze_historical_data()
                 Calls sub-modules:
                   baseline_stats.build_all_sku_stats()
                   correlation.build_correlation_matrix()
                   relationship.relationship_score_matrix()

    Step 2 → data_analyzer.compute_data_driven_lift()
                 Observed promos → elasticity model → fallback
                 Baseline is TREND-ADJUSTED (OLS projected + R²-blended with mean)

    Step 3 → relationship.classify_relationships()
                 Hybrid: HuggingFace embeddings + Token Jaccard + SequenceMatcher
                 (falls back to sklearn TF-IDF or built-in if no sentence-transformers)

    Step 4 → data_analyzer.compute_data_driven_cannibalization()
                 Uses correlation matrix + classified relationships

    Step 5 → profit.compute_profit()

    Step 6 → risk.compute_risk()

    Step 7 → verdict.compute_verdict()

    Step 8 → llm_client.generate_narrative()

No Streamlit imports. Pure Python.
"""

from engine.data_analyzer import (
    analyze_historical_data,
    compute_data_driven_lift,
    compute_data_driven_cannibalization,
)
from engine.relationship import classify_relationships, get_embedding_backend
from engine.profit       import compute_profit
from engine.risk         import compute_risk
from engine.verdict      import compute_verdict
from engine.llm_client   import generate_narrative


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _enrich_catalog(catalog: list[dict], summary) -> list[dict]:
    """
    Merge live per-SKU stats from the DataSummary into the catalog dicts.
    Ensures price and margin come from historical data rather than
    potentially stale catalog defaults.
    """
    enriched = []
    for sku in catalog:
        sid = str(sku.get("id", ""))
        sd  = summary.skus.get(sid)
        enriched.append({
            **sku,
            "price":  float(sd.full_price if sd else sku.get("price", 10.0)),
            "margin": float(sd.margin_pct  if sd else sku.get("margin", 40.0)),
        })
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_engine(inputs: dict) -> dict:
    """
    Run the full promotion analysis pipeline.

    Args:
        inputs (dict):
            api_key       (str)            : OpenAI API key (optional)
            historical_df (pd.DataFrame)   : Raw historical sales data
            promoted_sku  (dict)           : The SKU being promoted
            catalog_skus  (list[dict])     : Full catalog including promoted SKU
            discount_pct  (float)          : Proposed discount percentage
            duration_days (int)            : Promotion duration in days
            context_text  (str)            : Optional free-text context

    Returns:
        dict : Combined outputs from all pipeline steps + LLM narrative.
    """
    promoted    = inputs["promoted_sku"]
    catalog     = inputs["catalog_skus"]
    hist_df     = inputs["historical_df"]
    promoted_id = str(promoted["id"])
    discount_pct = float(inputs["discount_pct"])

    # ── Step 1: Analyse historical data ──────────────────────────────────────
    summary = analyze_historical_data(hist_df)

    sku_data = summary.skus.get(promoted_id)

    # Robust fallback logic
    full_price = float(sku_data.full_price if sku_data else promoted.get("price", 10.0))
    margin_pct = float(sku_data.margin_pct  if sku_data else promoted.get("margin", 40.0))

    # USE YOUR PRE-DEFINED CATEGORY DIRECTLY
    # If SKU data exists, use the actual category from the CSV
    category_type = sku_data.category 

    
    # ── Step 2: Data-driven lift ──────────────────────────────────────────────
    lift_result = compute_data_driven_lift(
        summary      = summary,
        sku_id       = promoted_id,
        discount_pct = discount_pct,
        target_week  = inputs.get("target_week"),   # optional; None = next week
    )
    baseline_units    = lift_result["baseline_units"]
    incremental_units = lift_result["incremental_units"]

    # ── Step 3: Classify relationships (hybrid embedding) ────────────────────
    #
    # classify_relationships() uses the best available embedding backend:
    #   1. HuggingFace sentence-transformers 
    #
    # This overwrites any pre-set relationship labels in the catalog,
    # ensuring the cannibalization step always uses fresh hybrid scores.
    enriched_catalog = _enrich_catalog(catalog, summary)
    rels = classify_relationships(enriched_catalog, promoted_id)
    for sku, rel in zip(enriched_catalog, rels):
        sku["relationship"] = rel

    # ── Step 4: Cannibalization (correlation + relationships) ─────────────────
    canni_result = compute_data_driven_cannibalization(
        summary           = summary,
        promoted_sku_id   = promoted_id,
        catalog_skus      = enriched_catalog,
        incremental_units = incremental_units,
    )

    # ── Feature 5: Post-promo dip ─────────────────────────────────────────────
    dip_data = summary.post_promo_dip.get(promoted_id, {})
    post_promo_dip_units = float(dip_data.get("total_dip_units", 0.0))

    # ── Feature 6: Competitor price from historical data ──────────────────────
    competitor_price: float | None = None
    if "competitor_price" in hist_df.columns:
        sku_hist = hist_df[hist_df["sku"] == promoted_id]
        comp_vals = sku_hist["competitor_price"].dropna()
        if not comp_vals.empty:
            competitor_price = round(float(comp_vals.mean()), 2)

    # ── Feature 7: Promo fatigue ──────────────────────────────────────────────
    fatigue_data = summary.promo_fatigue.get(promoted_id, {})
    # Get fatigue for the requested discount bucket
    from engine.baseline_stats import bucket_discount
    discount_bucket = bucket_discount(discount_pct / 100)
    fatigue_for_bucket = fatigue_data.get(discount_bucket, {})

    # ── Step 5: Profit ────────────────────────────────────────────────────────
    profit_result = compute_profit(
        baseline_units            = baseline_units,
        incremental_units         = incremental_units,
        full_price                = full_price,
        discount_pct              = discount_pct,
        margin_pct                = margin_pct,
        cannibalized_revenue_loss = canni_result["total_cannibalized_revenue"],
        post_promo_dip_units      = post_promo_dip_units,
    )

    # ── Step 6: Risk ──────────────────────────────────────────────────────────
    risk_result = compute_risk(
        discount_pct     = discount_pct,
        margin_pct       = margin_pct,
        cannibalized_pct = canni_result["cannibalized_pct"],
        context_text     = inputs.get("context_text", ""),
        category_type    = category_type,
        full_price       = full_price,
        competitor_price = competitor_price,
    )

    # ── Step 7: Verdict ───────────────────────────────────────────────────────
    verdict_result = compute_verdict(
        net_profit_delta = profit_result["net_profit_delta"],
        overall_risk     = risk_result["overall_risk"],
    )

    # ── Assemble combined output ───────────────────────────────────────────────
    combined: dict = {
        "category_type":       category_type,
        "product_name":        promoted.get("name", promoted_id),
        "discount_pct":        discount_pct,
        "discounted_price":    round(full_price * (1 - discount_pct / 100), 2),
        "duration_days":       inputs["duration_days"],
        "context_text":        inputs.get("context_text", ""),
        "baseline_units":          baseline_units,
        "lift_confidence":         lift_result.get("confidence", "medium"),
        "lift_method":             lift_result.get("method", "model"),
        "baseline_units_raw":      lift_result.get("baseline_units_raw", baseline_units),
        "trend_slope":             lift_result.get("trend_slope", 0.0),
        "trend_r2":                lift_result.get("trend_r2", 0.0),
        "recency_weighted_lift":   lift_result.get("recency_weighted_lift"),
        "raw_avg_lift":            lift_result.get("raw_avg_lift"),
        "recency_effect":          lift_result.get("recency_effect", 0.0),
        "lift_snr":                lift_result.get("lift_snr", 0.0),
        "confidence_before_snr":   lift_result.get("confidence_before_snr", "medium"),
        # Feature 4 — Seasonality
        "seasonality_index":                lift_result.get("seasonality_index", 1.0),
        "seasonality_adjusted_baseline":    lift_result.get("seasonality_adjusted_baseline", baseline_units),
        # Feature 5 — Post-promo dip
        "post_promo_dip":          dip_data,
        "post_promo_dip_units":    post_promo_dip_units,
        # Feature 6 — Competitor price
        "competitor_price":        competitor_price,
        # Feature 7 — Promo fatigue
        "promo_fatigue_all_buckets": fatigue_data,
        "promo_fatigue_this_bucket": fatigue_for_bucket,
        "promo_fatigue_flag":        fatigue_for_bucket.get("fatigue_flag", False),
        "embedding_backend":       get_embedding_backend(),
        "data_skus":           list(summary.skus.keys()),
        **lift_result,
        **canni_result,
        **profit_result,
        **risk_result,
        **verdict_result,
    }

    # ── Step 8: LLM narrative ─────────────────────────────────────────────────
    combined["narrative"] = generate_narrative(
        api_key = inputs.get("api_key", ""),
        outputs = combined,
    )

    return combined