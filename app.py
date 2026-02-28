"""
app.py
------
PriceSense AI — single-page side-by-side layout.
Left: inputs. Right: live results.

Run with:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="PriceSense AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "**PriceSense AI** — Promotion Intelligence for Mid-Market Retailers",
    },
)

# ── Session state defaults ──────────────────────────────────────────────────
DEFAULTS: dict = {
    "api_key":            "",
    "retailer_name":      "",
    "historical_df":      None,
    "catalog_skus":       [],
    "duration_days":      7,
    "promotion_question": "",
    "result":             None,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## 📊 PriceSense AI")
st.caption("Promotion Intelligence · Should I run this promotion?")
st.divider()

# ── Single page ──────────────────────────────────────────────────────────────
from ui.screen_main import render_home
render_home()