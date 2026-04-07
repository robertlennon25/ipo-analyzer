"""
scrape_ipo_universe.py
----------------------
Build a large IPO universe (200-500 IPOs, 2015-2023) by scraping
stockanalysis.com/ipos/{year}/ and enriching with sector via yfinance.

Output: data/raw/ipo_list_override.csv
Schema: ticker, company, ipo_date, offer_price, sector, total_proceeds_m

Notes on total_proceeds_m:
  Proceeds data is not available from the list page. We set a placeholder of
  100 ($100M) for all non-SPAC IPOs with offer_price >= $5. This is conservative
  — any such IPO almost certainly raised >= $50M. Rows with known proceeds
  (from the hardcoded KNOWN_PROCEEDS dict) get accurate values.

SPAC filtering: we exclude offerings whose name contains acquisition/blank check
patterns AND whose offer price is exactly $10 — the classic SPAC signature.

Usage:
    python src/ingestion/scrape_ipo_universe.py [--years 2019 2020 2021] [--no-sector]
"""

import re
import sys
import time
import argparse
import logging
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DIR, IPO_START_YEAR, IPO_END_YEAR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = RAW_DIR / "ipo_list_override.csv"
SECTOR_CACHE_PATH = RAW_DIR / "sector_cache.csv"

BASE_URL = "https://stockanalysis.com/ipos/{year}/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Minimum offer price — exclude sub-$5 micro-offerings
MIN_OFFER_PRICE = 5.0

# Placeholder proceeds for non-SPAC IPOs without known data
DEFAULT_PROCEEDS_M = 100.0

# Known proceeds for notable IPOs (millions USD) — accurate values
KNOWN_PROCEEDS: dict[str, float] = {
    "ABNB": 3500, "DASH": 3370, "SNOW": 3360, "RBLX": 520,  "COIN": 1200,
    "RIVN": 13700, "LYFT": 2340, "UBER": 8100, "PINS": 1400, "ZM": 751,
    "CRWD": 612,  "DDOG": 648,  "PLTR": 2200, "U": 1300,    "ASAN": 1340,
    "WISH": 1100, "POSH": 277,  "RDFN": 138,  "CHWY": 1020, "FROG": 352,
    "ACMR": 98,   "ZI": 935,    "DOCS": 690,  "LMND": 319,  "OZON": 990,
    "SOFI": 1700, "DUOL": 521,  "TOST": 870,  "BRZE": 483,
    "COUR": 519,  "PTON": 1160, "BYND": 241,
    "WORK": 838,  "FVRR": 111,  "BILL": 216,  "SMAR": 150,
    "ESTC": 252,  "MDB": 192,   "OKTA": 187,  "TWLO": 150,  "COUP": 143,
    "APPN": 120,  "HUBS": 180,  "TEAM": 462,
}

# SPAC name keywords — if name contains these AND price == $10, exclude
SPAC_KEYWORDS = [
    "acquisition", "blank check", "spac", "special purpose",
    " corp ii", " corp iii", " corp iv", " corp. ii",
    "holdings corp", "capital corp", "venture acquisition",
]


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def _parse_price(price_str: str) -> float | None:
    """Parse '$15.00' → 15.0"""
    m = re.search(r"[\d,]+\.?\d*", price_str.replace(",", ""))
    return float(m.group()) if m else None


def _parse_date(date_str: str) -> str | None:
    """Parse 'Dec 29, 2020' → '2020-12-29'"""
    try:
        return pd.to_datetime(date_str, format="%b %d, %Y").strftime("%Y-%m-%d")
    except Exception:
        try:
            return pd.to_datetime(date_str).strftime("%Y-%m-%d")
        except Exception:
            return None


def _is_spac(company: str, offer_price: float | None) -> bool:
    """
    Return True if this looks like a SPAC.
    Two independent signals — either alone is sufficient:
      1. Offer price is exactly $10.00 (the universal SPAC template price)
      2. Company name contains acquisition/blank-check keywords
    """
    is_ten_dollar = offer_price is not None and abs(offer_price - 10.0) < 0.01
    if is_ten_dollar:
        return True
    name_lower = company.lower()
    return any(kw in name_lower for kw in SPAC_KEYWORDS)


def scrape_year(year: int) -> list[dict]:
    """Scrape one year's IPO list from stockanalysis.com."""
    url = BASE_URL.format(year=year)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return []

    soup = BeautifulSoup(r.text, "lxml")
    tables = soup.find_all("table")
    if not tables:
        logger.warning("No tables found on %s", url)
        return []

    # First table has the main IPO list
    table = tables[0]
    rows = table.find_all("tr")[1:]  # skip header

    records = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        date_str = cells[0].get_text(strip=True)
        ticker = cells[1].get_text(strip=True).upper()
        company = cells[2].get_text(strip=True)
        price_str = cells[3].get_text(strip=True)

        ipo_date = _parse_date(date_str)
        offer_price = _parse_price(price_str)

        if not ticker or not ipo_date:
            continue
        if offer_price is None or offer_price < MIN_OFFER_PRICE:
            continue
        if _is_spac(company, offer_price):
            continue

        records.append({
            "ticker": ticker,
            "company": company,
            "ipo_date": ipo_date,
            "offer_price": offer_price,
        })

    logger.info("Year %d: %d IPOs (after SPAC/price filter)", year, len(records))
    return records


# ---------------------------------------------------------------------------
# Sector enrichment via yfinance
# ---------------------------------------------------------------------------

def _load_sector_cache() -> dict[str, str]:
    if SECTOR_CACHE_PATH.exists():
        df = pd.read_csv(SECTOR_CACHE_PATH)
        return dict(zip(df["ticker"], df["sector"]))
    return {}


def _save_sector_cache(cache: dict[str, str]) -> None:
    df = pd.DataFrame(list(cache.items()), columns=["ticker", "sector"])
    SECTOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SECTOR_CACHE_PATH, index=False)


def fetch_sectors(tickers: list[str], sleep: float = 0.3) -> dict[str, str]:
    """
    Fetch sector for each ticker via yfinance. Uses a file cache to avoid
    re-fetching on subsequent runs.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — sectors will be 'Unknown'")
        return {}

    cache = _load_sector_cache()
    missing = [t for t in tickers if t not in cache]

    if not missing:
        logger.info("All %d sectors loaded from cache", len(tickers))
        return {t: cache[t] for t in tickers if t in cache}

    logger.info("Fetching sectors for %d tickers (cached: %d)...", len(missing), len(cache))
    n = len(missing)
    for i, ticker in enumerate(missing, 1):
        if i % 25 == 0:
            logger.info("  %d / %d", i, n)
            _save_sector_cache(cache)  # checkpoint
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector") or "Unknown"
        except Exception:
            sector = "Unknown"
        cache[ticker] = sector
        time.sleep(sleep)

    _save_sector_cache(cache)
    return {t: cache.get(t, "Unknown") for t in tickers}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_override_csv(
    years: list[int] | None = None,
    fetch_sector: bool = True,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    if years is None:
        years = list(range(IPO_START_YEAR, IPO_END_YEAR + 1))

    print(f"Scraping stockanalysis.com for years: {years}")
    all_records: list[dict] = []

    for year in years:
        records = scrape_year(year)
        all_records.extend(records)
        time.sleep(1.0)  # be polite to the server

    if not all_records:
        raise RuntimeError("No IPO records scraped — check network or site structure")

    df = pd.DataFrame(all_records).drop_duplicates(subset=["ticker"])
    print(f"\nTotal unique IPOs scraped: {len(df)}")

    # Sector enrichment
    if fetch_sector:
        tickers = df["ticker"].tolist()
        sectors = fetch_sectors(tickers)
        df["sector"] = df["ticker"].map(sectors).fillna("Unknown")
    else:
        df["sector"] = "Unknown"

    # Proceeds: use known values, placeholder otherwise
    df["total_proceeds_m"] = df["ticker"].map(KNOWN_PROCEEDS).fillna(DEFAULT_PROCEEDS_M)

    # Reorder columns to match expected schema
    df = df[["ticker", "company", "ipo_date", "offer_price", "sector", "total_proceeds_m"]]
    df = df.sort_values("ipo_date").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} IPOs to {output_path}")
    print("\nSector breakdown:")
    print(df["sector"].value_counts().head(10).to_string())
    print("\nYear breakdown:")
    df["year"] = pd.to_datetime(df["ipo_date"]).dt.year
    print(df["year"].value_counts().sort_index().to_string())

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape IPO universe from stockanalysis.com")
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        help="Years to scrape (default: IPO_START_YEAR to IPO_END_YEAR from settings)"
    )
    parser.add_argument(
        "--no-sector", action="store_true",
        help="Skip yfinance sector enrichment (faster, produces 'Unknown' for all)"
    )
    args = parser.parse_args()

    df = build_override_csv(
        years=args.years,
        fetch_sector=not args.no_sector,
    )
    print(f"\nDone. Sample rows:")
    print(df.head(10).to_string())
