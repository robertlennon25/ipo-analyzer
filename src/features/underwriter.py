"""
underwriter.py
--------------
Extract underwriter information from IPO filing sections.

Extraction strategy (in priority order):
  1. Search all section text for "book-running manager" / underwriter context windows.
     The summary section often contains lead-underwriter disclosure near cover page language.
  2. Fall back to first 80 KB of raw filing HTML (cover page table).

Features produced per IPO:
  lead_underwriter_name     (str)   canonical name of the lead underwriter
  lead_underwriter_tier     (int)   1 / 2 / 3
  num_tier1_underwriters    (int)   count of Tier-1 banks detected
  num_tier2_underwriters    (int)   count of Tier-2 banks detected
  num_underwriters_total    (int)   all unique banks detected
  has_tier1_underwriter     (int)   binary flag
  underwriter_tier_strength (float) sum(4 - tier) for each bank found
                                    (Tier-1 → 3 pts, Tier-2 → 2 pts, Tier-3 → 1 pt)

Output: data/processed/underwriter_features.csv
"""

import re
import sys
import json
import logging
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, RAW_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SECTIONS_DIR = PROCESSED_DIR / "sections"
FILINGS_DIR  = RAW_DIR / "filings"
OUTPUT_PATH  = PROCESSED_DIR / "underwriter_features.csv"

# ---------------------------------------------------------------------------
# Normalization map: list of (regex_pattern, canonical_name).
# More specific / longer patterns first to avoid partial matches.
# Edit freely — add new aliases or reorder.
# ---------------------------------------------------------------------------
NORMALIZATION_PATTERNS: list[tuple[str, str]] = [
    # Goldman Sachs
    (r"goldman[,\s]+sachs", "Goldman Sachs"),
    # Morgan Stanley
    (r"morgan\s+stanley", "Morgan Stanley"),
    # J.P. Morgan / JPMorgan
    (r"j\.?p\.?\s*morgan|jpmorgan", "J.P. Morgan"),
    # Bank of America / Merrill Lynch / BofA
    (r"merrill\s+lynch|bofa\s+securities|bank\s+of\s+america", "Bank of America"),
    # Citigroup / Citi (avoid matching standalone "citi" in "Citibank" or city names)
    (r"citigroup|citi\s+(?:global|securities)", "Citigroup"),
    # Deutsche Bank
    (r"deutsche\s+bank", "Deutsche Bank"),
    # Barclays
    (r"barclays", "Barclays"),
    # Credit Suisse
    (r"credit\s+suisse", "Credit Suisse"),
    # UBS
    (r"ubs\s+(?:securities|investment|ag)", "UBS"),
    # Wells Fargo
    (r"wells\s+fargo", "Wells Fargo"),
    # RBC Capital
    (r"rbc\s+capital|royal\s+bank\s+of\s+canada", "RBC Capital"),
    # HSBC
    (r"hsbc\s+(?:securities|global)", "HSBC"),
    # Nomura
    (r"nomura\s+(?:securities|global)", "Nomura"),
    # Mizuho
    (r"mizuho\s+(?:securities|financial)", "Mizuho"),
    # Jefferies
    (r"jefferies", "Jefferies"),
    # Piper Sandler (formerly Piper Jaffray)
    (r"piper\s+sandler|piper\s+jaffray", "Piper Sandler"),
    # Cowen
    (r"cowen\s+(?:and|&)\s+company|cowen\s+inc\b", "Cowen"),
    # Stifel — trailing \b avoids matching "Stifel Nicolaus" differently
    (r"stifel\b", "Stifel"),
    # William Blair
    (r"william\s+blair", "William Blair"),
    # Needham
    (r"needham\s+(?:&|and)\s+company|needham\s+&\s+co", "Needham"),
    # Canaccord Genuity
    (r"canaccord\s+genuity|canaccord\s+corp", "Canaccord Genuity"),
    # KeyBanc
    (r"keybanc\s+capital|key\s+capital\s+markets", "KeyBanc"),
    # Truist
    (r"truist\s+securities", "Truist"),
    # Raymond James
    (r"raymond\s+james", "Raymond James"),
    # Oppenheimer
    (r"oppenheimer\s+(?:&|and|\s)", "Oppenheimer"),
    # Baird (R.W. Baird / Robert W. Baird)
    (r"r\.?w\.?\s*baird|robert\s+w\.?\s*baird|baird\s+(?:&|and|co)", "Baird"),
    # Guggenheim
    (r"guggenheim\s+(?:securities|partners)", "Guggenheim"),
    # Evercore
    (r"evercore\s+(?:isi|inc|group|partners|securities)", "Evercore"),
    # Lazard
    (r"lazard\s+(?:freres|capital|ltd)", "Lazard"),
    # Craig-Hallum
    (r"craig[-\s]hallum", "Craig-Hallum"),
    # Wedbush
    (r"wedbush\s+(?:securities|morgan)", "Wedbush"),
    # Imperial Capital
    (r"imperial\s+capital", "Imperial Capital"),
    # Janney Montgomery Scott
    (r"janney\s+montgomery", "Janney"),
    # Loop Capital
    (r"loop\s+capital", "Loop Capital"),
    # Leerink Partners / SVB Leerink
    (r"leerink|svb\s+leerink", "Leerink Partners"),
    # BTIG
    (r"\bBTIG\b", "BTIG"),
    # Lake Street
    (r"lake\s+street\s+capital", "Lake Street"),
]

# ---------------------------------------------------------------------------
# Tier map: canonical name → tier.  Default is Tier 3.
# Edit this dict to reclassify banks.
# ---------------------------------------------------------------------------
TIER_MAP: dict[str, int] = {
    # Tier 1 — dominant in global IPO league tables
    "Goldman Sachs":   1,
    "Morgan Stanley":  1,
    "J.P. Morgan":     1,
    "Bank of America": 1,
    "Citigroup":       1,
    "Deutsche Bank":   1,
    "Barclays":        1,
    "Credit Suisse":   1,
    "UBS":             1,
    "Wells Fargo":     1,
    "RBC Capital":     1,
    "HSBC":            1,
    "Nomura":          1,
    "Mizuho":          1,
    # Tier 2 — recognized mid-tier, strong in growth/tech IPOs
    "Jefferies":          2,
    "Piper Sandler":      2,
    "Cowen":              2,
    "Stifel":             2,
    "William Blair":      2,
    "Needham":            2,
    "Canaccord Genuity":  2,
    "KeyBanc":            2,
    "Truist":             2,
    "Raymond James":      2,
    "Oppenheimer":        2,
    "Baird":              2,
    "Guggenheim":         2,
    "Evercore":           2,
    "Lazard":             2,
    # Tier 3: all others — assigned by default in get_tier()
}

# Regex patterns that signal "the following are the lead/book-running managers"
LEAD_CONTEXT_PATTERNS: list[str] = [
    r"joint\s+book[-\s]running\s+managers?",
    r"book[-\s]running\s+managers?",
    r"lead\s+book[-\s]running\s+managers?",
    r"joint\s+lead\s+managers?",
    r"lead\s+managers?",
    r"lead\s+underwriters?",
    r"the\s+underwriters?\s+(?:named|listed|are)\b",
    r"underwritten\s+by",
    r"acting\s+as\s+(?:joint\s+)?(?:lead\s+)?(?:book[-\s]running\s+)?managers?",
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def get_tier(canonical_name: str) -> int:
    """Return tier for a canonical bank name (default Tier 3)."""
    return TIER_MAP.get(canonical_name, 3)


def _find_banks_in_text(text: str) -> list[str]:
    """Return list of unique canonical bank names found in text, in order of first appearance."""
    text_lower = text.lower()
    found: list[str] = []
    seen: set[str] = set()
    for pattern, canonical in NORMALIZATION_PATTERNS:
        if re.search(pattern, text_lower) and canonical not in seen:
            seen.add(canonical)
            found.append(canonical)
    return found


def _lead_banks_from_text(text: str) -> list[str]:
    """Search for underwriting context windows; return banks found in those windows."""
    text_lower = text.lower()
    for ctx_pattern in LEAD_CONTEXT_PATTERNS:
        match = re.search(ctx_pattern, text_lower)
        if match:
            window = text[match.end(): match.end() + 800]
            banks = _find_banks_in_text(window)
            if banks:
                return banks
    return []


def _extract_from_sections(sections: dict) -> dict:
    """Try to find underwriters from parsed section text."""
    full_text = " ".join(
        v for k, v in sections.items()
        if isinstance(v, str) and k not in {"ticker", "filing_type", "filing_date", "error"}
    )
    lead_banks = _lead_banks_from_text(full_text)
    all_banks = _find_banks_in_text(full_text)
    return {"lead_banks": lead_banks, "all_banks": all_banks}


def _extract_from_raw_html(ticker: str) -> dict:
    """Fall back to first 80 KB of raw filing HTML (cover page)."""
    ticker_dir = FILINGS_DIR / ticker
    if not ticker_dir.exists():
        return {"lead_banks": [], "all_banks": []}

    filing_files = sorted(ticker_dir.glob("*.html"))
    preferred = [f for f in filing_files if "424" in f.name]
    candidates = preferred if preferred else filing_files
    if not candidates:
        return {"lead_banks": [], "all_banks": []}

    try:
        raw = candidates[0].read_bytes()[:80_000].decode("utf-8", errors="ignore")
        from bs4 import BeautifulSoup
        text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    except Exception as exc:
        logger.warning("Raw HTML read failed for %s: %s", ticker, exc)
        return {"lead_banks": [], "all_banks": []}

    lead_banks = _lead_banks_from_text(text)
    all_banks = _find_banks_in_text(text)
    return {"lead_banks": lead_banks, "all_banks": all_banks}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_underwriter_features(ticker: str, sections: dict) -> dict:
    """Extract underwriter features for a single IPO. Returns flat feature dict."""
    result = _extract_from_sections(sections)

    # Fall back to raw HTML if sections yielded nothing
    if not result["all_banks"]:
        logger.debug("No banks in sections for %s — trying raw HTML", ticker)
        result = _extract_from_raw_html(ticker)

    all_banks = result["all_banks"]
    lead_banks = result["lead_banks"] if result["lead_banks"] else all_banks[:1]

    if not all_banks:
        logger.warning("No underwriters found for %s", ticker)
        return {
            "ticker": ticker,
            "lead_underwriter_name": None,
            "lead_underwriter_tier": None,
            "num_tier1_underwriters": 0,
            "num_tier2_underwriters": 0,
            "num_underwriters_total": 0,
            "has_tier1_underwriter": 0,
            "underwriter_tier_strength": 0.0,
        }

    # Lead = highest-tier (lowest tier number) bank from lead context
    lead_name = min(lead_banks, key=get_tier)
    lead_tier = get_tier(lead_name)
    tiers = [get_tier(b) for b in all_banks]

    return {
        "ticker": ticker,
        "lead_underwriter_name": lead_name,
        "lead_underwriter_tier": lead_tier,
        "num_tier1_underwriters": sum(1 for t in tiers if t == 1),
        "num_tier2_underwriters": sum(1 for t in tiers if t == 2),
        "num_underwriters_total": len(all_banks),
        "has_tier1_underwriter": int(any(t == 1 for t in tiers)),
        "underwriter_tier_strength": float(sum(4 - t for t in tiers)),
    }


def build_underwriter_features(
    sections_dir: Path = SECTIONS_DIR,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Process all section JSONs and save underwriter features CSV."""
    section_files = sorted(sections_dir.glob("*.json"))
    if not section_files:
        raise FileNotFoundError(f"No section JSON files in {sections_dir}. Run section_extractor.py first.")

    print(f"Extracting underwriter features for {len(section_files)} IPOs...")
    records = []
    found_count = 0

    for path in section_files:
        try:
            sections = json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path.name, exc)
            continue

        ticker = sections.get("ticker", path.stem)
        feat = extract_underwriter_features(ticker, sections)
        records.append(feat)
        if feat["lead_underwriter_name"]:
            found_count += 1

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}  ({len(df)} rows)")
    print(f"Underwriter identified: {found_count}/{len(df)} ({found_count/len(df):.0%})")
    print("\nLead underwriter tier distribution:")
    tier_counts = df["lead_underwriter_tier"].value_counts().sort_index()
    for tier, cnt in tier_counts.items():
        label = {1: "Tier 1 (major global IB)", 2: "Tier 2 (mid-tier)", 3: "Tier 3 (other)"}.get(int(tier), str(tier))
        print(f"  {label}: {cnt}")
    print(f"  Not found: {df['lead_underwriter_name'].isna().sum()}")
    return df


if __name__ == "__main__":
    build_underwriter_features()
