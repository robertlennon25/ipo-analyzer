"""
edgar_fetcher.py
----------------
Download S-1 and 424B4 filings from SEC EDGAR for a list of companies.

CIK lookup strategy (in order):
  1. EDGAR company search by ticker  (fastest, works for most US companies)
  2. EDGAR full-text search by company name (fallback for mismatches)

Filing preference: 424B4 (final prospectus) > S-1 (initial registration)

Output:
  data/raw/filings/{ticker}/*.html   raw filing HTML
  data/processed/filing_manifest.csv  per-filing metadata

Usage:
  python src/ingestion/edgar_fetcher.py              # all companies in universe
  python src/ingestion/edgar_fetcher.py --limit 20   # first 20 (for testing)
  python src/ingestion/edgar_fetcher.py --limit 5 --ticker SNOW CRWD DDOG
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DIR, PROCESSED_DIR, EDGAR_USER_AGENT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FILINGS_DIR = RAW_DIR / "filings"
FILINGS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = PROCESSED_DIR / "filing_manifest.csv"

HEADERS = {"User-Agent": EDGAR_USER_AGENT}
FILING_TYPES = ["424B4", "S-1"]          # preference order
RATE_LIMIT_SLEEP = 0.5                   # SEC allows ~10 req/s
MAX_FILINGS_PER_COMPANY = 3


# ---------------------------------------------------------------------------
# Retry / rate-limit helpers
# ---------------------------------------------------------------------------

def _get(url: str, *, timeout: int = 20, max_retries: int = 5) -> requests.Response:
    """
    GET with exponential backoff on 429 / 5xx.
    Raises requests.HTTPError on persistent failure.
    """
    delay = 2.0
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise
            logger.warning("Request error (%s), retry %d in %.0fs", exc, attempt + 1, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        if r.status_code == 429 or r.status_code >= 500:
            if attempt == max_retries - 1:
                r.raise_for_status()
            wait = delay + float(r.headers.get("Retry-After", 0))
            logger.warning("HTTP %d — backing off %.0fs (attempt %d)", r.status_code, wait, attempt + 1)
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue

        r.raise_for_status()
        return r

    raise RuntimeError(f"Exhausted retries for {url}")


# ---------------------------------------------------------------------------
# CIK lookup
# ---------------------------------------------------------------------------

def _cik_from_company_search(ticker: str) -> str | None:
    """
    Strategy 1: EDGAR company search by ticker symbol.
    Parses the Atom feed returned by browse-edgar.
    """
    url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?company=&CIK={ticker}&type=S-1&dateb=&owner=include"
        f"&count=5&search_text=&action=getcompany&output=atom"
    )
    try:
        r = _get(url)
        soup = BeautifulSoup(r.text, "xml")
        tag = soup.find("cik")
        if tag:
            return tag.text.strip()
    except Exception as exc:
        logger.debug("Company search failed for %s: %s", ticker, exc)
    return None


def _cik_from_fulltext_search(company_name: str) -> str | None:
    """
    Strategy 2: EDGAR full-text search by company name.
    Uses the EFTS (EDGAR Full-Text Search) API.
    """
    url = (
        f"https://efts.sec.gov/LATEST/search-index"
        f'?q="{requests.utils.quote(company_name)}"&forms=S-1'
    )
    try:
        r = _get(url)
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])
        if hits:
            entity = hits[0].get("_source", {})
            cik = entity.get("entity_id") or entity.get("cik")
            if cik:
                return str(int(cik)).zfill(10)
    except Exception as exc:
        logger.debug("Full-text search failed for '%s': %s", company_name, exc)
    return None


def get_cik(ticker: str, company_name: str) -> tuple[str | None, str]:
    """Resolve CIK using both strategies; return (zero-padded CIK or None, strategy)."""
    cik = _cik_from_company_search(ticker)
    time.sleep(RATE_LIMIT_SLEEP)
    if cik:
        return str(int(cik)).zfill(10), "ticker"

    cik = _cik_from_fulltext_search(company_name)
    time.sleep(RATE_LIMIT_SLEEP)
    if cik:
        return cik, "name search"

    return None, "not found"


# ---------------------------------------------------------------------------
# Filing discovery
# ---------------------------------------------------------------------------

def _parse_filing_page(page: dict) -> list[dict]:
    """Extract target filings from one submissions page dict."""
    forms        = page.get("form", [])
    dates        = page.get("filingDate", [])
    accessions   = page.get("accessionNumber", [])
    primary_docs = page.get("primaryDocument", [])

    results = []
    for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
        if form in FILING_TYPES:
            results.append({
                "filing_type": form,
                "filing_date": date,
                "accession_number": acc,
                "primary_document": doc,
            })
    return results


def get_filings_for_cik(cik: str) -> list[dict]:
    """
    Fetch filing list from EDGAR submissions API.
    Checks the main submissions JSON plus any older-filing archive pages.
    Returns filings sorted: 424B4 first, then S-1, newest-first within each type.
    """
    base_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        r = _get(base_url)
        data = r.json()
    except Exception as exc:
        logger.warning("  Submissions fetch failed for CIK %s: %s", cik, exc)
        return []

    results = _parse_filing_page(data.get("filings", {}).get("recent", {}))

    # Fetch additional archive pages (older filings) if S-1 not yet found
    has_s1 = any(f["filing_type"] in FILING_TYPES for f in results)
    if not has_s1:
        for file_entry in data.get("filings", {}).get("files", []):
            name = file_entry.get("name", "")
            if not name:
                continue
            url = f"https://data.sec.gov/submissions/{name}"
            try:
                r2 = _get(url)
                page_data = r2.json()
                results.extend(_parse_filing_page(page_data))
                time.sleep(RATE_LIMIT_SLEEP)
            except Exception as exc:
                logger.debug("  Could not fetch archive page %s: %s", name, exc)

    # Attach CIK
    for f in results:
        f["cik"] = cik

    # Sort: 424B4 before S-1, newest first within each type
    type_rank = {ft: i for i, ft in enumerate(FILING_TYPES)}
    results.sort(key=lambda x: x["filing_date"], reverse=True)
    results.sort(key=lambda x: type_rank.get(x["filing_type"], 99))

    return results[:MAX_FILINGS_PER_COMPANY]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_filing(cik: str, accession: str, doc: str, save_path: Path) -> tuple[bool, int]:
    """Download one filing document to disk. Returns (success, bytes_written)."""
    acc_clean = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
    try:
        r = _get(url, timeout=60)
        content = r.text
        save_path.write_text(content, encoding="utf-8", errors="replace")
        return True, len(content.encode("utf-8"))
    except Exception as exc:
        logger.warning("  Download failed: %s — %s", url, exc)
        return False, 0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def fetch_filings_for_universe(
    ipo_df: pd.DataFrame,
    limit: int | None = None,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch S-1 / 424B4 filings for each company in ipo_df.

    Args:
        ipo_df:  DataFrame with columns [ticker, company, ipo_date]
        limit:   If set, cap at N companies (useful for smoke tests)
        tickers: If set, restrict to these specific tickers
    """
    universe = ipo_df.copy()
    if tickers:
        universe = universe[universe["ticker"].isin(tickers)]
    if limit:
        universe = universe.head(limit)

    records = []
    total = len(universe)
    n_ok = n_cached = n_failed = n_skipped = 0

    def _tally() -> str:
        return f"ok:{n_ok} cached:{n_cached} failed:{n_failed} skipped:{n_skipped}"

    for i, (_, row) in enumerate(universe.iterrows(), 1):
        ticker  = row["ticker"]
        company = row["company"]
        ipo_date = str(row["ipo_date"])[:10]

        print(f"\n[{i}/{total}] {ticker} — {company}  ({_tally()})")

        ticker_dir = FILINGS_DIR / ticker
        ticker_dir.mkdir(exist_ok=True)

        # Skip if already downloaded
        existing = list(ticker_dir.glob("*.html")) + list(ticker_dir.glob("*.htm"))
        if existing:
            sizes = ", ".join(f"{f.stat().st_size // 1024}KB" for f in existing)
            print(f"  Cached: {len(existing)} file(s) [{sizes}] — skipping")
            for f in existing:
                ft = "424B4" if "424" in f.name else "S-1"
                records.append({
                    "ticker": ticker, "company": company,
                    "filing_type": ft, "filing_date": ipo_date,
                    "file_path": str(f), "status": "cached",
                })
            n_cached += len(existing)
            continue

        # Resolve CIK
        cik, strategy = get_cik(ticker, company)
        if not cik:
            print(f"  CIK not found (tried ticker + name search) — skipping")
            records.append({"ticker": ticker, "company": company, "status": "no_cik"})
            n_skipped += 1
            continue

        print(f"  CIK: {cik}  (via {strategy})")

        # Get filing list
        filings = get_filings_for_cik(cik)
        time.sleep(RATE_LIMIT_SLEEP)

        if not filings:
            print(f"  No S-1 or 424B4 found in EDGAR submissions")
            records.append({"ticker": ticker, "company": company, "status": "no_filing"})
            n_skipped += 1
            continue

        types_found = [f["filing_type"] for f in filings]
        print(f"  Found filings: {types_found}")

        for filing in filings:
            fname = (
                f"{filing['filing_type'].replace('/', '_')}"
                f"_{filing['filing_date']}"
                f"_{filing['accession_number'][:8]}.html"
            )
            save_path = ticker_dir / fname

            print(f"  Downloading {filing['filing_type']} ({filing['filing_date']}) ...", end="", flush=True)
            ok, nbytes = download_filing(cik, filing["accession_number"], filing["primary_document"], save_path)
            time.sleep(RATE_LIMIT_SLEEP)

            if ok:
                print(f" {nbytes // 1024:,}KB")
                n_ok += 1
            else:
                print(f" FAILED")
                n_failed += 1

            records.append({
                "ticker": ticker,
                "company": company,
                "filing_type": filing["filing_type"],
                "filing_date": filing["filing_date"],
                "accession_number": filing["accession_number"],
                "file_path": str(save_path) if ok else None,
                "status": "ok" if ok else "download_failed",
            })

    manifest = pd.DataFrame(records)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    print(f"\n{'='*50}")
    print(f"Done. Manifest saved: {MANIFEST_PATH}")
    print(f"  Downloaded: {n_ok}  |  Cached: {n_cached}  |  Failed: {n_failed}  |  Skipped (no CIK/filing): {n_skipped}")
    return manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download S-1/424B4 filings from EDGAR")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process only the first N companies (for testing)"
    )
    parser.add_argument(
        "--ticker", nargs="+", default=None, metavar="TICK",
        help="Restrict to specific ticker(s), e.g. --ticker SNOW CRWD"
    )
    args = parser.parse_args()

    universe_path = PROCESSED_DIR / "ipo_universe.csv"
    if not universe_path.exists():
        print("ipo_universe.csv not found — run ipo_list.py first")
        sys.exit(1)

    universe = pd.read_csv(universe_path, parse_dates=["ipo_date"])
    print(f"Universe: {len(universe)} IPOs")

    manifest = fetch_filings_for_universe(
        universe,
        limit=args.limit,
        tickers=args.ticker,
    )

    if "status" in manifest.columns:
        ok = (manifest["status"] == "ok").sum()
        print(f"Successfully downloaded: {ok} filings")
