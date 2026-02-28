# PriceSense AI

**Promotion Intelligence for Mid-Market Retailers**

> *"Should I run this promotion?"* — answered with your own sales data.

PriceSense AI is a Streamlit app that takes your historical sales CSV and a plain-English question, runs an 8-step analytical pipeline, and returns a data-backed verdict: **Run It**, **Run with Changes**, or **Don't Run** — with a full breakdown of why.

---

## Demo

Ask a question like:

> *"Should we run 25% off on Salted Pistachios 16oz next week?"*

Get back:

- Projected unit lift (from observed promos or price elasticity model)
- Net profit delta after margin compression, cannibalization, and post-promo hangover
- Cross-SKU catalog impact (which products bleed, which get a halo)
- 5-signal risk score including competitor pricing and promo fatigue
- Plain-English analyst brief via GPT-4o

---

## Quick Start

```bash
git clone https://github.com/your-username/pricesense-ai
cd pricesense-ai

pip install -r requirements.txt

# Set your OpenAI key (required for intent parsing + analyst brief)
cp .envexample .env
# Edit .env and add your key

streamlit run app.py
```

A bundled sample dataset (`data/extended_historical_data.csv`, 8 SKUs × 104 weeks) loads automatically so you can try the app immediately without uploading anything.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | GPT-4o for question parsing and analyst brief |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | HuggingFace model for SKU relationship classification |

Create a `.env` file in the project root (see `.envexample`):

```env
OPENAI_API_KEY=sk-proj-...
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

> **Without an API key**, the app still runs — the analyst brief falls back to a rule-based template and question parsing defaults to the first SKU at 25% off.

---

## Data Format

Upload a CSV with these columns:

| Column | Type | Description |
|---|---|---|
| `sku` | str | SKU identifier (e.g. `PIS16`) |
| `name` | str | Product display name |
| `category` | str | Product category (e.g. `Nuts`, `Beverage`) |
| `week` | int | Week number (0-based, sequential) |
| `price` | float | Selling price that week |
| `cost` | float | Unit cost that week |
| `promo_flag` | int | `1` = promotion week, `0` = baseline week |
| `discount` | float | Discount fraction (e.g. `0.25` for 25% off) |
| `units_sold` | int | Units sold that week |
| `competitor_price` | float | *(optional)* Competitor's price — enables competitor risk signal |

Regenerate or customise the sample data:

```bash
python data/generate_data.py
```

---

## How the Engine Works

The analysis runs an 8-step pipeline on every query:

```
Historical CSV
     │
     ▼
Step 1  analyze_historical_data()
        Per-SKU: trend-adjusted baseline, price elasticity (OLS),
        recency-weighted promo lift table, Pearson cross-SKU correlation,
        seasonality index, post-promo dip, promo fatigue
     │
     ▼
Step 2  compute_data_driven_lift()
        Lift source priority:
          1. Observed promos in matching discount bucket (≥3 obs, recency-weighted)
          2. OLS price elasticity model
          3. Fallback (discount × 1.2)
        Baseline is trend-adjusted then seasonality-scaled.
        Confidence downgraded if SNR (lift / demand std) < 1.0.
     │
     ▼
Step 3  classify_relationships()
        HuggingFace sentence-transformers + SequenceMatcher
        → variant / substitute / complement / unrelated
     │
     ▼
Step 4  compute_data_driven_cannibalization()
        Bleed rate = relationship base rate × Pearson correlation scalar
        Revenue impact computed per catalog SKU
     │
     ▼
Step 5  compute_profit()
        net_profit_delta = promo_profit
                         − baseline_profit
                         − cannibalization_loss
                         − post_promo_dip_loss
     │
     ▼
Step 6  compute_risk()
        5 signals → composite score → low / medium / high
          • Margin compression     (discount / margin ratio, per-category thresholds)
          • Cannibalization bleed  (% of lift lost to catalog)
          • Timing / context       (keyword scan of free-text context)
          • Competitor price gap   (your price vs competitor_price column)
          • Promo fatigue          (declining lift trend over successive runs)
     │
     ▼
Step 7  compute_verdict()
        profit_delta > 0 + overall_risk → RUN IT / RUN WITH CHANGES / DON'T RUN
     │
     ▼
Step 8  generate_narrative()
        GPT-4o analyst brief (3 bullet points, plain English)
        Falls back to rule-based template if no API key
```

### Signals computed from your data

| Signal | What it captures |
|---|---|
| **Trend-adjusted baseline** | OLS trend projected to promotion week, blended with flat mean via R² |
| **Recency-weighted lift** | Recent promos weighted more heavily (exponential decay, half-life ≈ 35 weeks) |
| **SNR confidence guard** | Lift confidence downgraded when lift < 1× demand std (signal inside noise) |
| **Seasonality index** | Week-of-year multiplier from non-promo weeks — peak weeks amplify ROI |
| **Post-promo dip** | Measured pantry-loading hangover in weeks t+1 and t+2, deducted from profit |
| **Competitor price gap** | Your price vs `competitor_price` column — adjusts timing risk |
| **Promo fatigue** | OLS slope across successive promo runs — flags declining lift effectiveness |
| **Cross-SKU correlation** | Pearson correlation scales cannibalization bleed rate per SKU pair |

---

## Project Structure

```
app.py                              ← Page config, session state, entry point
requirements.txt
.envexample

engine/
  orchestrator.py                   ← Runs the 8-step pipeline
  baseline_stats.py                 ← Per-SKU stats: trend, elasticity,
  │                                     seasonality, post-promo dip, promo fatigue
  data_analyzer.py                  ← DataSummary container, lift + cannibalization
  correlation.py                    ← Pearson cross-SKU correlation matrix
  relationship.py                   ← HuggingFace SKU relationship classifier
  profit.py                         ← Net profit delta calculation
  risk.py                           ← 5-signal composite risk score
  verdict.py                        ← Run It / Run with Changes / Don't Run
  llm_client.py                     ← GPT-4o narrative (template fallback)

ui/
  screen_main.py                    ← Layout + run_engine call (~180 lines)
  components/
    result_panel.py                 ← Assembles all result sub-components
    signals.py                      ← 7 signal insight renderers
    financials.py                   ← Catalog impact table + financials expander
  utils/
    catalog.py                      ← build_catalog_from_df()
    intent_parser.py                ← parse_question() via GPT-4o

config/
  settings.py                       ← API key + embedding model from env / .env

data/
  extended_historical_data.csv      ← Bundled sample (8 SKUs × 104 weeks)
  generate_data.py                  ← Synthetic data generator
```

---

## Deployment — Streamlit Community Cloud

No Docker required.

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/your-username/pricesense-ai.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your repo, branch `main`, entry point `app.py`
4. Open **Advanced settings → Secrets** and add:

```toml
OPENAI_API_KEY = "sk-proj-..."
```

5. Click **Deploy**

> The first boot takes 2–3 minutes while `sentence-transformers` downloads the embedding model (~90 MB). Subsequent boots are fast.

### 3. Update settings.py for Streamlit secrets

`config/settings.py` reads from env vars and `.env`. On Streamlit Cloud, secrets come through `st.secrets`. Add this check to `get_openai_api_key()`:

```python
def get_openai_api_key() -> str:
    try:
        import streamlit as st
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    _load_dotenv()
    return os.environ.get("OPENAI_API_KEY", "").strip()
```

---

## Known Limitations

| Limitation | Impact |
|---|---|
| Baseline is weekly, not daily | Can't model intra-week demand patterns |
| Seasonality uses week % 52 | Doesn't handle multi-year holiday shifts (e.g. Easter) |
| Post-promo dip is a 2-week window | Longer pantry-loading effects are underestimated |
| Fatigue detection needs ≥3 runs per bucket | New or rarely-promoted SKUs return no fatigue signal |
| Competitor price is a CSV column | No live price scraping — data is as fresh as your last upload |
| LLM narrative requires OpenAI key | ~$0.01 per analysis; falls back to template without key |

---

## License

MIT
