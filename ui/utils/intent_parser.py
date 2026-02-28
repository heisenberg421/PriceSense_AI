"""
ui/utils/intent_parser.py
--------------------------
Extracts SKU ID and discount % from a natural-language promotion question
using GPT-4o in JSON mode.
"""

from __future__ import annotations
import json
from openai import OpenAI
import streamlit as st


def parse_question(question: str, catalog: list[dict], api_key: str) -> dict:
    """
    Uses GPT-4o to extract the SKU ID and discount percentage from natural language.

    Returns dict with keys:
        sku_id       : str   — matched SKU id
        discount_pct : int   — discount percentage (default 25 if not specified)
    """
    if not api_key:
        st.error("OpenAI API Key is missing. Please configure it in settings.")
        return {"sku_id": None, "discount_pct": 25}

    sku_context = "\n".join([f"- ID: {s['id']} | Name: {s['name']}" for s in catalog])

    system_prompt = (
        "You are a retail operations assistant. Your task is to extract promotion details "
        "from a user's question. You will be provided with a list of available SKUs.\n\n"
        "Identify:\n"
        "1. The 'sku_id' that best matches the product mentioned.\n"
        "2. The 'discount_pct' as an integer (default to 25 if not specified).\n\n"
        "Constraint: You MUST respond with ONLY a valid JSON object. No prose."
    )
    user_prompt = f"Available SKUs:\n{sku_context}\n\nQuestion: {question}"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"LLM Parsing Error: {e}")
        return {"sku_id": catalog[0]["id"] if catalog else None, "discount_pct": 25}
