"""
ui/utils/catalog.py
-------------------
Builds the SKU catalog dict from a historical sales DataFrame.
"""

from __future__ import annotations
import pandas as pd


def build_catalog_from_df(df: pd.DataFrame) -> list[dict]:
    """
    Builds catalog using literal names and categories from the data file.
    No artificial display names are generated.
    """
    catalog, seen = [], set()
    df = df.copy()

    for sku_id, grp in df.groupby("sku"):
        if sku_id in seen:
            continue
        seen.add(sku_id)

        base_rows  = grp[grp["promo_flag"] == 0] if "promo_flag" in grp.columns else grp
        full_price = float(base_rows["price"].mean() if not base_rows.empty else grp["price"].max())
        unit_cost  = float(grp["cost"].mean()) if "cost" in grp.columns else 0.0
        margin_pct = round(((full_price - unit_cost) / full_price * 100) if full_price > 0 else 40.0, 1)

        real_name = str(grp["name"].iloc[0])            if "name"     in grp.columns else str(sku_id)
        category  = str(grp["category"].mode().iloc[0]) if "category" in grp.columns else "General"

        catalog.append({
            "id":           str(sku_id),
            "name":         real_name,
            "price":        round(full_price, 2),
            "margin":       margin_pct,
            "category":     category,
            "relationship": "unrelated",
        })

    return sorted(catalog, key=lambda x: (x["category"], x["id"]))
