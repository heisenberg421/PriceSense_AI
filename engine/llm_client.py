"""
engine/llm_client.py
--------------------
Step 8 of the engine pipeline.

Calls the OpenAI API to generate a plain-language analyst brief.
Synchronized with predefined categories and variant/substitute relationship logic.
"""

import os

_SYSTEM_PROMPT = """You are a senior retail pricing analyst at a top-tier consulting firm.
You receive structured data about a proposed retail promotion and write a clear,
concise recommendation brief for a retail merchant — not a data scientist.

Your output must be:
- Written in plain business English (no jargon)
- Exactly 3 short bullet points, each 1-2 sentences long max 40 words, covering:
    Point 1: What the data shows (key numbers, unit lift, and net profit)
    Point 2: The cannibalization profile (specifically mentioning Variants or Substitutes)
    Point 3: A concrete, specific action recommendation (based on the SYSTEM VERDICT)
- Honest about the fact that these are model estimates, not guarantees
- Direct — the merchant needs to make a decision today

Do not use headers. Do not use markdown formatting. Just write 3 clean points separated by blank lines.
"""

def _build_prompt(outputs: dict) -> str:
    """Constructs the data-heavy string for the LLM to interpret."""
    
    # Enrich the SKU breakdown for the AI
    # It now distinguishes between size variants and competitive substitutes
    sku_list = ", ".join(
        "{name} ({rel}, {dir}{units} units)".format(
            name=s["name"],
            rel=s["relationship"],
            dir="\u2212" if s["direction"] == "loss" else "+",
            units=s["lost_units"],
        )
        for s in outputs.get("sku_breakdown", [])
    ) or "None"

    suggestion_line = (
        f"SUGGESTED CHANGE: {outputs['suggested_change']}"
        if outputs.get("suggested_change")
        else ""
    )

    method_note = ""
    if outputs.get("lift_method") == "observed":
        method_note = f"Lift estimate based on {outputs.get('lift_confidence', 'medium')}-confidence observed historical promotions."
    elif outputs.get("lift_method") == "elasticity_model":
        method_note = "Lift estimate based on price elasticity model fitted to historical data."

    # [Image of a retail executive dashboard summary]
    
    return f"""RETAILER: {outputs.get('retailer_name', 'Retailer')} | CATEGORY: {outputs['category_type']}

PROPOSED PROMOTION:
  Product:   {outputs['product_name']}
  Discount:  {outputs['discount_pct']}% off (promo price: ${outputs['discounted_price']})
  Duration:  {outputs['duration_days']} day(s)
  Context:   "{outputs.get('context_text') or 'None provided'}"

ENGINE OUTPUTS (data-driven):
  Baseline weekly units:    {outputs['baseline_units']}
  Projected lift:           {outputs['lift_pct']}% (+{outputs['incremental_units']} units)
  {method_note}
  Cannibalization:          {outputs['cannibalized_pct']}% of lift bleeds to other SKUs
  Net profit delta:         ${outputs['net_profit_delta']:,.2f}

RISK SIGNALS:
  Margin compression:       {outputs['margin_risk']}
  Cannibalization risk:     {outputs['cannibalization_risk']}
  Timing / context risk:    {outputs['timing_risk']}
  Overall risk:             {outputs['overall_risk']}

CATALOG IMPACT (Bleed Details): {sku_list}

SYSTEM VERDICT: {outputs['label']}
{suggestion_line}

Write the 3-point analyst brief now."""


def _fallback_narrative(outputs: dict) -> str:
    """Template narrative when no API key is provided or on error."""
    verdict = outputs.get("label", "Run with Changes")
    lift    = outputs.get("lift_pct", 0)
    canni   = outputs.get("cannibalized_pct", 0)
    profit  = outputs.get("net_profit_delta", 0)
    disc    = outputs.get("discount_pct", 0)

    p1 = (
        f"A {disc}% discount on {outputs.get('product_name', 'this SKU')} is projected "
        f"to drive {lift:.1f}% unit lift, adding {outputs.get('incremental_units', 0)} "
        f"incremental units. This results in a net profit impact of "
        f"{'${:,.2f}'.format(abs(profit))} {'gain' if profit >= 0 else 'loss'}."
    )

    p2 = (
        f"Cannibalization is at {canni:.1f}%, with notable bleed into "
        f"{'related products' if outputs.get('sku_breakdown') else 'the broader catalog'}. "
        f"The {outputs.get('overall_risk', 'medium')} overall risk score reflects "
        f"{outputs.get('margin_risk', 'medium')} margin pressure."
    )

    if outputs.get("suggested_change"):
        p3 = f"Recommendation: {outputs['suggested_change']}"
    else:
        p3 = f"Verdict: {verdict}. Monitor category performance closely to ensure the lift offsets the margin compression."

    return f"{p1}\n\n{p2}\n\n{p3}"


def generate_narrative(api_key: str, outputs: dict) -> str:
    """
    Generate analyst narrative via OpenAI GPT-4o.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")

    if not key:
        return _fallback_narrative(outputs)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        # Image of Large Language Model training for retail analytics
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.4,
            max_tokens=500,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_prompt(outputs)},
            ],
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        return _fallback_narrative(outputs) + f"\n\n_(Note: AI narrative unavailable — {str(exc)[:100]})_"