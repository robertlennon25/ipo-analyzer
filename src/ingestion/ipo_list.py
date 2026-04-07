"""
ipo_list.py
-----------
Build the IPO universe: ticker, IPO date, offer price, sector, offer size.

Sources (in order of preference):
1. Local CSV override (data/raw/ipo_list_override.csv) — manually curated
2. Scraped from site:stockanalysis.com/ipos/  (free, decent coverage)
3. Fallback: a hardcoded sample for development/testing

Output: data/processed/ipo_universe.csv
Columns: ticker, company, ipo_date, offer_price, sector, shares_offered, total_proceeds_m
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DIR, PROCESSED_DIR, IPO_START_YEAR, IPO_END_YEAR, MIN_OFFER_SIZE_M


OVERRIDE_PATH = RAW_DIR / "ipo_list_override.csv"
OUTPUT_PATH = PROCESSED_DIR / "ipo_universe.csv"

# --- Development sample (used as fallback) ---
SAMPLE_IPOS = [
    {"ticker": "ABNB", "company": "Airbnb", "ipo_date": "2020-12-10", "offer_price": 68.0, "sector": "Consumer Discretionary", "total_proceeds_m": 3500},
    {"ticker": "DASH", "company": "DoorDash", "ipo_date": "2020-12-09", "offer_price": 102.0, "sector": "Technology", "total_proceeds_m": 3370},
    {"ticker": "SNOW", "company": "Snowflake", "ipo_date": "2020-09-16", "offer_price": 120.0, "sector": "Technology", "total_proceeds_m": 3360},
    {"ticker": "RBLX", "company": "Roblox", "ipo_date": "2021-03-10", "offer_price": 45.0, "sector": "Technology", "total_proceeds_m": 520},
    {"ticker": "COIN", "company": "Coinbase", "ipo_date": "2021-04-14", "offer_price": 250.0, "sector": "Financials", "total_proceeds_m": 1200},
    {"ticker": "RIVN", "company": "Rivian", "ipo_date": "2021-11-10", "offer_price": 78.0, "sector": "Consumer Discretionary", "total_proceeds_m": 13700},
    {"ticker": "LYFT", "company": "Lyft", "ipo_date": "2019-03-29", "offer_price": 72.0, "sector": "Technology", "total_proceeds_m": 2340},
    {"ticker": "UBER", "company": "Uber", "ipo_date": "2019-05-10", "offer_price": 45.0, "sector": "Technology", "total_proceeds_m": 8100},
    {"ticker": "PINS", "company": "Pinterest", "ipo_date": "2019-04-18", "offer_price": 19.0, "sector": "Technology", "total_proceeds_m": 1400},
    {"ticker": "ZM",   "company": "Zoom Video", "ipo_date": "2019-04-18", "offer_price": 36.0, "sector": "Technology", "total_proceeds_m": 751},
    {"ticker": "CRWD", "company": "CrowdStrike", "ipo_date": "2019-06-12", "offer_price": 34.0, "sector": "Technology", "total_proceeds_m": 612},
    {"ticker": "DDOG", "company": "Datadog", "ipo_date": "2019-09-19", "offer_price": 27.0, "sector": "Technology", "total_proceeds_m": 648},
]


def load_override() -> pd.DataFrame | None:
    if OVERRIDE_PATH.exists():
        print(f"Loading override IPO list from {OVERRIDE_PATH}")
        return pd.read_csv(OVERRIDE_PATH, parse_dates=["ipo_date"])
    return None


def load_sample() -> pd.DataFrame:
    print("Using hardcoded sample IPO list (development fallback)")
    df = pd.DataFrame(SAMPLE_IPOS)
    df["ipo_date"] = pd.to_datetime(df["ipo_date"])
    return df


def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    original = len(df)
    df = df[df["ipo_date"].dt.year.between(IPO_START_YEAR, IPO_END_YEAR)]
    df = df[df["total_proceeds_m"] >= MIN_OFFER_SIZE_M]
    df = df.dropna(subset=["ticker", "ipo_date", "offer_price"])
    df = df.drop_duplicates(subset=["ticker"])
    df = df.sort_values("ipo_date").reset_index(drop=True)
    print(f"Filtered {original} → {len(df)} IPOs (date range + min size)")
    return df


def build_universe() -> pd.DataFrame:
    df = load_override()
    if df is None:
        df = load_sample()

    df = filter_universe(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} IPOs to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    df = build_universe()
    print(df.head(10).to_string())
