"""
price_fetcher.py
----------------
Fetch post-IPO price data and compute returns at all target windows.

Handles edge cases:
- IPO open price vs offer price discrepancy
- Ticker changes / delistings
- Missing data for shorter-lived companies

Output: data/processed/returns.csv
Columns: ticker, offer_price, ipo_date, open_1d, close_1d, ret_1d, ret_1w, ret_1m, ret_6m, ret_1y, label_1m, label_6m, label_1y
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, CACHE_DIR, RETURN_WINDOWS

OUTPUT_PATH = PROCESSED_DIR / "returns.csv"
CACHE_PATH = CACHE_DIR / "price_cache.csv"


def fetch_price_history(ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame | None:
    """Download OHLCV data for a ticker. Returns None if unavailable."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start_date, end=end_date, auto_adjust=True)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"  yfinance error for {ticker}: {e}")
        return None


def compute_returns(price_df: pd.DataFrame, ipo_date: str, offer_price: float) -> dict:
    """
    Compute returns from IPO date at all target windows.
    
    Uses Day 1 open as the "actual entry price" (more realistic than offer price
    for public market investors), but also records offer-to-close pop.
    """
    results = {}
    ipo_dt = pd.Timestamp(ipo_date)

    # yfinance returns a tz-aware index; normalize to tz-naive dates for comparison
    index = price_df.index
    if hasattr(index, "tz") and index.tz is not None:
        index = index.tz_localize(None)
    price_df = price_df.copy()
    price_df.index = index

    # Find first trading day on or after IPO date
    trading_days = price_df.index[price_df.index >= ipo_dt]
    if len(trading_days) == 0:
        return results

    first_day = trading_days[0]
    day1_open = price_df.loc[first_day, "Open"]
    day1_close = price_df.loc[first_day, "Close"]

    results["ipo_trading_date"] = str(first_day.date())
    results["open_1d"] = round(day1_open, 4)
    results["close_1d"] = round(day1_close, 4)
    results["ret_offer_to_open"] = round((day1_open - offer_price) / offer_price, 4)
    results["ret_1d"] = round((day1_close - day1_open) / day1_open, 4)
    results["ret_offer_to_close_1d"] = round((day1_close - offer_price) / offer_price, 4)

    # Subsequent window returns (from Day 1 open)
    for window_name, n_days in RETURN_WINDOWS.items():
        if window_name == "1d":
            continue
        future_days = trading_days[trading_days > first_day]
        if len(future_days) >= n_days:
            target_day = future_days[n_days - 1]
            target_close = price_df.loc[target_day, "Close"]
            ret = (target_close - day1_open) / day1_open
            results[f"ret_{window_name}"] = round(ret, 4)
        else:
            results[f"ret_{window_name}"] = np.nan

    return results


def add_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary labels: 1 if return > 0, else 0."""
    for window in ["1d", "1w", "1m", "6m", "1y"]:
        col = f"ret_{window}"
        if col in df.columns:
            df[f"label_{window}"] = (df[col] > 0).astype(int)
    return df


def fetch_all_returns(ipo_df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """
    Main entry point. Fetch returns for all IPOs in universe.
    """
    # Load cache if available
    if use_cache and CACHE_PATH.exists():
        cached = pd.read_csv(CACHE_PATH)
        cached_tickers = set(cached["ticker"].tolist())
        print(f"Cache hit: {len(cached_tickers)} tickers already fetched")
    else:
        cached = pd.DataFrame()
        cached_tickers = set()

    new_records = []
    to_fetch = ipo_df[~ipo_df["ticker"].isin(cached_tickers)]
    print(f"Fetching returns for {len(to_fetch)} new tickers...")

    for _, row in to_fetch.iterrows():
        ticker = row["ticker"]
        ipo_date = str(row["ipo_date"])[:10]
        offer_price = float(row["offer_price"])

        print(f"  {ticker} (IPO: {ipo_date}, offer: ${offer_price})")

        # Fetch ~14 months of data to cover all windows
        prices = fetch_price_history(ticker, start_date=ipo_date)
        time.sleep(0.3)  # Be polite to yfinance

        record = {
            "ticker": ticker,
            "company": row.get("company", ""),
            "ipo_date": ipo_date,
            "offer_price": offer_price,
            "sector": row.get("sector", ""),
        }

        if prices is None or len(prices) < 2:
            print(f"    No price data available")
            record["status"] = "no_data"
        else:
            returns = compute_returns(prices, ipo_date, offer_price)
            record.update(returns)
            record["status"] = "ok"

        new_records.append(record)

    new_df = pd.DataFrame(new_records)

    # Combine with cache
    all_returns = pd.concat([cached, new_df], ignore_index=True) if not cached.empty else new_df

    # Save updated cache
    all_returns.to_csv(CACHE_PATH, index=False)

    # Add labels and save final output
    all_returns = add_binary_labels(all_returns)
    ok = all_returns[all_returns["status"] == "ok"]
    ok.to_csv(OUTPUT_PATH, index=False)

    print(f"\nReturns computed for {len(ok)}/{len(all_returns)} tickers")
    print(f"Saved to {OUTPUT_PATH}")

    # Quick summary
    for window in ["1d", "1w", "1m", "6m", "1y"]:
        col = f"ret_{window}"
        if col in ok.columns:
            valid = ok[col].dropna()
            if len(valid) > 0:
                pos_rate = (valid > 0).mean()
                print(f"  {window}: n={len(valid)}, median={valid.median():.1%}, positive rate={pos_rate:.1%}")

    return ok


if __name__ == "__main__":
    universe = pd.read_csv(PROCESSED_DIR / "ipo_universe.csv", parse_dates=["ipo_date"])
    returns_df = fetch_all_returns(universe)
    print(returns_df.head())
