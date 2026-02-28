import pandas as pd
import numpy as np

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Config for Full Catalog — Grouped for Robust Relationship Logic
# ──────────────────────────────────────────────────────────────────────────────
CATALOG_CONFIG = {
    # Beverage Category
    "WAT_SPK_6":  {"name": "Sparkling Water 6-pack",  "price": 4.99,  "cost": 2.10, "cat": "Beverage",  "base": 800},
    "WAT_SPK_12": {"name": "Sparkling Water 12-pack", "price": 7.49,  "cost": 3.50, "cat": "Beverage",  "base": 650},
    "WAT_SPK_24": {"name": "Sparkling Water 24-pack", "price": 12.99, "cost": 6.80, "cat": "Beverage",  "base": 400},
    "WAT_STL_12": {"name": "Still Water 12-pack",     "price": 5.99,  "cost": 2.40, "cat": "Beverage",  "base": 900},
    "WAT_FLV_12": {"name": "Flavored Water 12-pack",  "price": 8.49,  "cost": 3.90, "cat": "Beverage",  "base": 550},
    
    # Personal Care Category
    "LOT_HND_SM": {"name": "Hand Lotion Small",       "price": 3.49,  "cost": 1.20, "cat": "Personal Care", "base": 300},
    "LOT_HND_MD": {"name": "Hand Lotion Medium",      "price": 6.99,  "cost": 2.80, "cat": "Personal Care", "base": 250},
    "LOT_BDY_LG": {"name": "Body Lotion Large",       "price": 11.99, "cost": 5.10, "cat": "Personal Care", "base": 200},
    
    # Dairy & Cereal Category
    "YOG_GRK_8":  {"name": "Greek Yogurt 8oz",        "price": 1.79,  "cost": 0.85, "cat": "Dairy",     "base": 1200},
    "YOG_REG_8":  {"name": "Regular Yogurt 8oz",      "price": 1.29,  "cost": 0.55, "cat": "Dairy",     "base": 1500},
    "MILK_OAT_64":{"name": "Oat Milk 64oz",           "price": 4.99,  "cost": 2.30, "cat": "Dairy",     "base": 700},
    "GRA_ORG_12": {"name": "Granola Organic 12oz",    "price": 5.49,  "cost": 2.60, "cat": "Snacks",    "base": 450},

    # Health & Vitamins Category
    "VIT_D_60":   {"name": "Vitamin D 60ct",          "price": 9.99,  "cost": 3.50, "cat": "Health",    "base": 180},
    "VIT_CAL_90": {"name": "Calcium + 90ct",          "price": 14.49, "cost": 5.20, "cat": "Health",    "base": 150},
    "VIT_MULTI":  {"name": "Multivitamin 100ct",      "price": 19.99, "cost": 8.00, "cat": "Health",    "base": 220},

    # Nuts & Snacks Category (Primary Test Group)
    "ALM8":       {"name": "Roasted Almonds 8oz",     "price": 5.50,  "cost": 3.40, "cat": "Nuts",      "base": 660},
    "ALM16":      {"name": "Roasted Almonds 16oz",    "price": 9.00,  "cost": 5.20, "cat": "Nuts",      "base": 500},
    "ALM32":      {"name": "Roasted Almonds 32oz",    "price": 16.00, "cost": 9.50, "cat": "Nuts",      "base": 445},
    "PIS8":       {"name": "Salted Pistachios 8oz",   "price": 6.00,  "cost": 3.80, "cat": "Nuts",      "base": 710},
    "PIS16":      {"name": "Salted Pistachios 16oz",  "price": 11.00, "cost": 6.50, "cat": "Nuts",      "base": 420},
    "PIS32":      {"name": "Salted Pistachios 32oz",  "price": 18.00, "cost": 11.00,"cat": "Nuts",      "base": 395},
    "MIXNUT16":   {"name": "Mixed Nuts Variety 16oz", "price": 9.50,  "cost": 5.50, "cat": "Nuts",      "base": 495},
}

WEEKS = 12

def generate_data():
    all_rows = []
    for sku, cfg in CATALOG_CONFIG.items():
        # 1-2 promo weeks for a 10-20% max frequency
        promo_weeks = np.random.choice(range(WEEKS), size=np.random.choice([1, 2]), replace=False)
        
        for wk in range(WEEKS):
            is_promo = wk in promo_weeks
            discount = np.random.choice([0.15, 0.20]) if is_promo else 0.0
            price = round(cfg["price"] * (1 - discount), 2)
            
            # Unit lift logic for 12-week snapshots
            lift = (1.45 + discount * 3.8) if is_promo else 1.0
            noise = np.random.normal(0, cfg["base"] * 0.05)
            units = int(cfg["base"] * lift + noise)
            
            all_rows.append({
                "sku": sku,
                "name": cfg["name"],
                "category": cfg["cat"],
                "week": wk + 1,
                "price": price,
                "cost": cfg["cost"],
                "promo_flag": 1 if is_promo else 0,
                "discount": discount,
                "units_sold": max(5, units)
            })
            
    df = pd.DataFrame(all_rows)
    df.to_csv("data\extended_historical_data.csv", index=False)
    print(f"Dataset generated with {len(df)} rows across {len(CATALOG_CONFIG)} SKUs.")

if __name__ == "__main__":
    generate_data()