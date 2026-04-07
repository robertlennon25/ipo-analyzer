# TODO

## Immediate — Run the Pipeline

The code is complete. Nothing runs end-to-end yet because only 2 filings (CRWD, SNOW) have been downloaded.

### Step 1: Download more filings
```bash
python src/ingestion/edgar_fetcher.py --limit 20
```
This is the bottleneck. EDGAR rate-limits to ~10 req/s; 20 companies takes ~5 minutes.
Need **≥ 30** companies for train.py cross-validation to work.

### Step 2: Fetch sector data (optional but improves M3)
```bash
python src/ingestion/scrape_ipo_universe.py   # ~10 min for 700 tickers via yfinance
```

### Step 3: Run the rest of the pipeline
```bash
python src/ingestion/price_fetcher.py
python src/parsing/section_extractor.py
python src/features/handcrafted.py
python src/features/multiples.py
python src/features/market_context.py
pip install sentence-transformers
python src/features/embeddings.py
python src/modeling/train.py
python src/modeling/evaluate.py
```

### Step 4: View results
```bash
cat data/processed/evaluation_report.md
streamlit run app/streamlit_app.py
```

---

## Remaining Code Tasks

### Task 6 — Polish `app/streamlit_app.py` (low priority until pipeline runs)
1. Add **"Run Pipeline"** button — executes ingestion scripts with live stdout
2. Add **"Predict New IPO"** page — paste prospectus URL or upload text, get prediction
3. Make IPO Explorer table sortable by return columns
4. Add return comparison chart: 1D/1W/1M/6M/1Y for selected IPO vs cohort median
5. Add scatter plot: sentiment_compound vs ret_1m, colored by sector

---

## Known Issues / Risks

| Issue | Severity | Notes |
|-------|----------|-------|
| `total_proceeds_m = 100` placeholder | Low | Most IPOs have placeholder value; doesn't affect modeling, only the $50M filter which all pass |
| Sector = "Unknown" for all 700 IPOs | Medium | Run `scrape_ipo_universe.py` without `--no-sector` to fix; affects sector ETF features in M3 |
| Market price cache only covers 2019-02 → 2020-09 | Medium | Will auto-extend when full universe runs |
| `EDGAR_USER_AGENT` in settings.py is placeholder | High | **Must update** before running edgar_fetcher.py or EDGAR will reject requests |
| train.py needs ≥ 30 samples | High | Run edgar_fetcher with at least 30-40 companies |
| sentence-transformers not installed in venv | Medium | `pip install sentence-transformers` before running embeddings.py |

---

## Research Notes

- Primary question: **does text add alpha over fundamentals alone?** Compare M1 vs M2 vs M3.
- Use temporal train/test split (by year) not random split — avoids year-level confounding from `is_hot_ipo_year`
- With ~300 IPOs, expect noisy results; 200+ is the minimum for meaningful signal detection
- SHAP values in evaluate.py will show which features drive predictions most
