# Claude Code Prompt: IPO Language & Aftermarket Performance Analyzer

Paste this prompt into Claude Code to continue building this project.

---

## PROMPT

I'm building an end-to-end research system called the **IPO Language & Aftermarket Performance Analyzer**. The project tests whether language in IPO filings (SEC S-1 / 424B4) contains measurable signals that predict post-IPO stock performance.

The project scaffold already exists with the following structure:

```
ipo-analyzer/
├── config/settings.py          # Constants, paths, keyword lists
├── data/
│   ├── raw/filings/            # EDGAR HTML filings (per ticker)
│   ├── processed/              # CSVs, section JSONs, model results
│   └── cache/                  # Price data, embedding cache
├── src/
│   ├── ingestion/
│   │   ├── ipo_list.py         # Build IPO universe ✅
│   │   ├── edgar_fetcher.py    # Download S-1/424B4 from EDGAR ✅
│   │   └── price_fetcher.py    # Fetch post-IPO returns via yfinance ✅
│   ├── parsing/
│   │   └── section_extractor.py # Extract sections from HTML filings ✅
│   ├── features/
│   │   ├── handcrafted.py      # Sentiment, uncertainty, keyword features ✅
│   │   ├── embeddings.py       # Sentence-transformer embeddings ✅
│   │   └── multiples.py        # MISSING — needs to be built
│   └── modeling/
│       ├── train.py            # M1/M2/M3 training loop ✅
│       └── evaluate.py         # MISSING — SHAP + detailed eval
├── app/streamlit_app.py        # Streamlit MVP UI ✅
└── requirements.txt            ✅
```

### Your tasks (in priority order):

---

**TASK 1: Build `src/features/multiples.py`**

This module should extract structured financial features from the S-1 filing text/HTML for the M2 and M3 model variants. Extract:

**From the filing text (regex + heuristic parsing):**
- Revenue (most recent fiscal year, and prior year for YoY growth)
- Gross profit / gross margin
- Net income / net loss
- EBITDA (if mentioned)
- Cash and cash equivalents
- Total assets

**From the IPO universe CSV (`data/processed/ipo_universe.csv`):**
- Offer size (total_proceeds_m) — already present
- Sector — already present

**Derived / engineered features:**
- Revenue growth YoY (%)
- Gross margin (%)
- Whether the company is profitable (binary)
- Net loss as % of revenue
- Proceeds as % of implied valuation (if calculable)
- Cash burn proxy

**From filing text (keyword-based):**
- % of proceeds going to insiders (selling stockholders) vs primary (company)
- Whether price range was revised upward before pricing (search for "amended" + price mentions)
- Number of risk factors (count numbered items in risk section)

Save output to `data/processed/multiples_features.csv`.

Note: Financial table parsing in raw HTML is messy. Use BeautifulSoup to find `<table>` tags, then apply regex to find dollar amounts near keywords like "revenue", "net loss", "gross profit". Fail gracefully — many fields will be NaN for many companies. That's fine.

---

**TASK 2: Build `src/modeling/evaluate.py`**

A standalone evaluation module that:
1. Loads trained models from `data/processed/models/`
2. Loads model results JSON from `data/processed/model_results.json`
3. Generates SHAP summary plots for each variant (save as PNGs to `data/processed/plots/`)
4. Produces a clean Markdown summary report (`data/processed/evaluation_report.md`) with:
   - Model comparison table (ROC-AUC, accuracy, n_samples)
   - Top 10 features per variant
   - Interpretation: does text add signal over fundamentals alone?
   - Notable finding callouts (e.g., "uncertainty_density is the #1 text predictor")

---

**TASK 3: Expand the IPO universe**

The current `ipo_list.py` has a hardcoded sample of ~12 IPOs. We need a real dataset.

Build a scraper or data loader that pulls a larger IPO list (ideally 200–500 IPOs, 2015–2023) from one of:
- `stockanalysis.com/ipos/` (scrapeable, decent coverage)
- A static CSV from a known source like `Renaissance Capital` or similar
- As fallback: generate a larger synthetic-but-realistic list from known IPO databases

The output should match the schema: `ticker, company, ipo_date, offer_price, sector, total_proceeds_m`

Save to `data/raw/ipo_list_override.csv` so it's auto-picked up by `ipo_list.py`.

---

**TASK 4: Fix and harden the EDGAR fetcher**

The current `edgar_fetcher.py` uses the EDGAR submissions API but CIK lookup is fragile. Improve it:

1. Try EDGAR company search first: `https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK={ticker}&type=S-1`
2. Fall back to EDGAR full-text search: `https://efts.sec.gov/LATEST/search-index?q="{company_name}"&forms=S-1`
3. For each company, prefer the **424B4** (final prospectus) over S-1 when both are available
4. Add a `--limit N` CLI argument to test with N companies before full run
5. Handle rate limiting gracefully (exponential backoff on 429s)

---

**TASK 5: Add market regime features**

In `src/features/` create a new file `market_context.py` that fetches and computes:
- VIX level on the IPO date (use yfinance ticker `^VIX`)
- S&P 500 trailing 30-day return on IPO date (ticker `^GSPC`)
- Sector ETF trailing 30-day return (map sectors to ETFs: XLK, XLF, XLV, etc.)
- Number of other IPOs in the same calendar month (hot market proxy — compute from ipo_universe.csv)
- Whether it's a "hot IPO year" (2020/2021 vs 2018/2019 etc.)

These go into M2 and M3 as additional structured features.

---

**TASK 6: Polish the Streamlit app**

The app at `app/streamlit_app.py` is functional but needs polish:

1. Add a **"Run Pipeline"** button on the Overview page that executes the ingestion scripts in order and shows live stdout
2. Add a **"Predict New IPO"** page where users can paste a prospectus URL or upload text, and get a model prediction
3. Make the IPO Explorer table sortable by return columns
4. Add a return comparison chart: bar chart of 1D / 1W / 1M / 6M / 1Y returns for a selected IPO vs the cohort median
5. Add a scatter plot: sentiment_compound vs ret_1m, colored by sector

---

### Important implementation notes:

- **Python 3.10+**, use type hints throughout
- Always use `config/settings.py` paths — never hardcode paths
- Cache expensive operations (price fetches, embeddings) aggressively
- Add `if __name__ == "__main__":` blocks to every module so they're runnable standalone
- When parsing financial tables from HTML, log warnings for failed extractions but never crash
- Keep the sample size problem in mind: with ~300 IPOs, avoid overfitting — use cross-validation everywhere, report std alongside mean
- The primary research question is: **does text signal add alpha over fundamentals?** Keep this framing central in the report and UI

### To run the full pipeline:
```bash
pip install -r requirements.txt
python src/ingestion/ipo_list.py
python src/ingestion/edgar_fetcher.py --limit 20   # test with 20 first
python src/ingestion/price_fetcher.py
python src/parsing/section_extractor.py
python src/features/handcrafted.py
python src/features/embeddings.py
python src/features/multiples.py
python src/features/market_context.py
python src/modeling/train.py
python src/modeling/evaluate.py
streamlit run app/streamlit_app.py
```

Start with Task 1 (`multiples.py`) as it unlocks the M2/M3 model variants which are the heart of the experiment.
