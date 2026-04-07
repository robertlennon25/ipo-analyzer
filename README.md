# IPO Language & Aftermarket Performance Analyzer

Research system that tests whether language in IPO filings (SEC S-1 / 424B4) predicts post-IPO stock performance. Primary question: **does text signal add alpha over fundamentals?**

## Results (Run 001 — April 2026)

Target: 1-month binary return (`label_1m`), n ≈ 425 IPOs, 5-fold CV

| Variant | Features | LR ROC-AUC | XGB ROC-AUC |
|---------|----------|-----------|------------|
| M1 — text only | 401 (NLP + embeddings) | **0.612 ± 0.029** | 0.601 ± 0.030 |
| M2 — fundamentals only | 18 (financials + market) | 0.535 ± 0.075 | 0.560 ± 0.052 |
| M3 — combined | 419 | **0.615 ± 0.038** | 0.605 ± 0.032 |

Text features alone (M1) outperform structured financials (M2). M3 adds a small increment over M1, suggesting financials contribute marginal signal beyond language. Full details and SHAP plots in [`results_tracker.md`](results_tracker.md).

## Model Variants

| Variant | Features | Research Question |
|---------|----------|-------------------|
| M1 | Handcrafted NLP + `all-MiniLM-L6-v2` embeddings | Pure language signal |
| M2 | Revenue, margins, proceeds + VIX/S&P/sector ETF | Pure fundamentals signal |
| M3 | M1 + M2 | Does text add alpha over numbers? |

Each variant is trained with Logistic Regression (interpretable baseline) and XGBoost (performance). Evaluated via stratified 5-fold CV, reporting ROC-AUC and accuracy with std.

## Setup

**Prerequisites:** Python 3.10+, SEC EDGAR account (free — just needs a name/email as User-Agent)

```bash
git clone <repo-url>
cd ipo-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers
```

Set your EDGAR user agent in `config/settings.py`:
```python
EDGAR_USER_AGENT = "Your Name your@email.com"
```

## Running the Pipeline

```bash
# 1. Build IPO universe with sectors (~4 min)
python src/ingestion/scrape_ipo_universe.py
python src/ingestion/ipo_list.py

# 2. Fetch price returns (~5 min, cached)
python src/ingestion/price_fetcher.py

# 3. Download filings from EDGAR
python src/ingestion/edgar_fetcher.py --limit 50   # test; drop --limit for full run (~3-5 hrs)

# 4. Market context features (can run alongside step 3)
python src/features/market_context.py

# 5. Extract sections + compute features
python src/parsing/section_extractor.py
python src/features/handcrafted.py
python src/features/multiples.py
python src/features/embeddings.py

# 6. Train and evaluate
python src/modeling/train.py
python src/modeling/evaluate.py
```

Results are written to `data/processed/evaluation_report.md` and `data/processed/plots/`.

## Data Sources

| Source | What | How |
|--------|------|-----|
| [stockanalysis.com](https://stockanalysis.com/ipos/) | IPO list 2019–2023 | Scraped via `scrape_ipo_universe.py` |
| SEC EDGAR | S-1 / 424B4 filing HTML | `edgar_fetcher.py` via EDGAR submissions API |
| yfinance | Post-IPO price history, sector | `price_fetcher.py`, `scrape_ipo_universe.py` |
| yfinance | VIX, S&P 500, sector ETFs | `market_context.py` |

## Project Structure

```
config/settings.py          — all paths, constants, keyword lists
src/ingestion/              — data collection (IPO list, EDGAR, prices)
src/parsing/                — HTML → structured section JSON
src/features/               — feature engineering (NLP, financials, market)
src/modeling/               — train.py, evaluate.py
app/streamlit_app.py        — interactive explorer (WIP)
data/raw/                   — ipo_list_override.csv, sector_cache.csv, filings/
data/processed/             — features CSVs, model results, SHAP plots
data/cache/                 — price cache, embeddings cache
results_tracker.md          — per-run model comparison log
```

## Notes

- `total_proceeds_m` is a placeholder ($100M) for most IPOs — stockanalysis.com doesn't publish proceeds in list view
- ~185/700 IPOs have unknown sector (delisted companies not found by yfinance)
- train.py requires ≥ 30 samples; recommend ≥ 300 filings for reliable results
- Use temporal train/test splits (not random) to avoid year-level confounding from `is_hot_ipo_year`
