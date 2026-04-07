# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does
Research system testing whether language in IPO filings (SEC S-1 / 424B4) predicts post-IPO stock performance. Primary question: **does text signal add alpha over fundamentals?**

Three model variants:
- **M1** — text only (handcrafted NLP features + sentence-transformer embeddings)
- **M2** — structured financials only (revenue, margins, proceeds, sector)
- **M3** — M1 + M2 + market context (VIX, S&P momentum, sector ETF, IPO volume)

---

## Setup & Commands

### Environment
```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers   # not in requirements.txt; needed for embeddings.py
```

### Full pipeline (run in this order)

```bash
# 1. Build IPO universe with sector enrichment (~4 min for 700 tickers via yfinance)
python src/ingestion/scrape_ipo_universe.py   # scrapes stockanalysis.com + fetches sectors
python src/ingestion/ipo_list.py              # filters → data/processed/ipo_universe.csv

# 2. Fetch price returns (fast, ~5 min for 700 tickers via yfinance, results cached)
python src/ingestion/price_fetcher.py

# 3. Download filings from EDGAR (slow — rate-limited to ~10 req/s)
python src/ingestion/edgar_fetcher.py --limit 50   # test run; drop --limit for full ~700 (3-5 hrs)

# 4. Market context features (no dependency on filings — can run alongside step 3)
python src/features/market_context.py

# 5. Extract text sections from downloaded filings (skips already-processed tickers)
python src/parsing/section_extractor.py

# 6. Compute features (all read from sections/ or filing HTML)
python src/features/handcrafted.py
python src/features/multiples.py
python src/features/embeddings.py

# 7. Train and evaluate
python src/modeling/train.py
python src/modeling/evaluate.py

# 8. View results
cat data/processed/evaluation_report.md
streamlit run app/streamlit_app.py
```

### Re-running after more filings are downloaded
Steps 1, 2, 4 do not need to be re-run. Only run:
```bash
python src/ingestion/edgar_fetcher.py          # picks up where it left off (skips cached tickers)
python src/parsing/section_extractor.py        # skips already-processed tickers
python src/features/handcrafted.py
python src/features/multiples.py
python src/features/embeddings.py
python src/modeling/train.py
python src/modeling/evaluate.py
```

---

## Architecture

### Data flow
```
scrape_ipo_universe.py
  → data/raw/ipo_list_override.csv (701 IPOs)
ipo_list.py
  → data/processed/ipo_universe.csv (filtered)
edgar_fetcher.py
  → data/raw/filings/{TICKER}/*.html + data/processed/filing_manifest.csv
price_fetcher.py
  → data/processed/returns.csv (per-window returns + binary labels)
section_extractor.py
  → data/processed/sections/{ticker}.json
handcrafted.py / embeddings.py / multiples.py / market_context.py
  → data/processed/{feature_set}_features.csv  (embeddings → data/cache/embeddings.npz)
train.py
  → data/processed/models/*.pkl + data/processed/model_results.json
evaluate.py
  → data/processed/plots/shap_*.png + data/processed/evaluation_report.md
```

### Module responsibilities
| Module | Role |
|--------|------|
| `config/settings.py` | Single source of truth for all paths and constants |
| `src/ingestion/edgar_fetcher.py` | CIK lookup, 424B4 preference over S-1, backoff, archive page fallback |
| `src/parsing/section_extractor.py` | Extracts named sections (summary, risk factors, business, etc.) from filing HTML |
| `src/features/handcrafted.py` | VADER sentiment, uncertainty/growth/profitability keyword densities, readability |
| `src/features/embeddings.py` | Sentence-transformer embeddings (`all-MiniLM-L6-v2`) per ticker, cached in `.npz` |
| `src/features/multiples.py` | Financial features from filing HTML via BS4+regex (revenue, margins, proceeds) |
| `src/features/market_context.py` | VIX, S&P500 trailing return, sector ETF, IPO volume — all as-of IPO date |
| `src/modeling/train.py` | Assembles M1/M2/M3 feature matrices; trains LogisticRegression + XGBoost; CV evaluation |
| `src/modeling/evaluate.py` | SHAP plots + Markdown evaluation report |
| `app/streamlit_app.py` | Explorer UI (unpolished; Task 6 pending per TODO.md) |

---

## Key Conventions
- Always import paths from `config/settings.py` — never hardcode
- Cache expensive ops in `data/cache/` (prices, embeddings, market data)
- Every module has `if __name__ == "__main__":` block and can be run standalone
- Log warnings for failed extractions, never crash — NaN is fine
- `CV_FOLDS=5`; always report std alongside mean
- `train.py` silently skips variants with < 30 samples

---

## Critical Setup Note
`EDGAR_USER_AGENT` in `config/settings.py` is a placeholder (`"your_name your_email@example.com"`). **Must be updated to a real name/email** before running `edgar_fetcher.py` — SEC will reject requests otherwise.

---

## Important Design Notes

### Information leakage in market_context.py
Market features use data on or before IPO date (no forward-looking bias), but two subtler risks exist:
1. **Year-level confounding**: `is_hot_ipo_year` encodes 2020/2021. Use temporal train/test splits (not random).
2. **Momentum autocorrelation**: `sp500_ret_30d` correlates with near-term returns. Run M2 vs M3 ablations to measure how much market context adds.

### total_proceeds_m placeholder
Scraper sets `total_proceeds_m = 100` for most IPOs (stockanalysis.com doesn't publish proceeds in list view). Known large IPOs have accurate values in `KNOWN_PROCEEDS` dict in the scraper. The $50M filter passes all rows since 100 ≥ 50.

### EDGAR older filings
The submissions API `recent` array only holds ~1000 filings. `edgar_fetcher.py` automatically fetches archive pages (`CIK{n}-submissions-001.json`) when the recent list has no target filings.

### Sector data
Sector = "Unknown" for all 700 IPOs until `scrape_ipo_universe.py` is re-run without `--no-sector`. This affects sector ETF features in M3.

---

## Remaining Work
See `TODO.md` for prioritized task list.
