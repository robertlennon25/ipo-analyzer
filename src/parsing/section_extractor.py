"""
section_extractor.py
--------------------
Parse raw S-1 / 424B4 HTML filings and extract key sections.

Sections targeted:
- Prospectus Summary
- Risk Factors
- Business
- Use of Proceeds
- Management's Discussion (MD&A)

Strategy:
1. Strip HTML → clean text
2. Find section boundaries by scanning for heading patterns
3. Extract text between boundaries
4. Truncate sections to max token budget

Output: data/processed/sections/{ticker}.json
"""

import re
import json
import warnings
from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import sys

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, RAW_DIR

SECTIONS_DIR = PROCESSED_DIR / "sections"
SECTIONS_DIR.mkdir(exist_ok=True)

# Max characters per section (to keep embeddings tractable)
MAX_SECTION_CHARS = 8000

# Section heading patterns (order matters — more specific first)
SECTION_PATTERNS = {
    "summary": [
        r"prospectus\s+summary",
        r"summary\s+of\s+the\s+offering",
        r"^summary$",
    ],
    "risk_factors": [
        r"risk\s+factors",
        r"risks\s+related\s+to",
    ],
    "business": [
        r"^business$",
        r"our\s+business",
        r"business\s+overview",
    ],
    "use_of_proceeds": [
        r"use\s+of\s+proceeds",
    ],
    "mda": [
        r"management.s\s+discussion",
        r"management.s\s+discussion\s+and\s+analysis",
        r"MD&A",
    ],
}

# Compiled patterns for efficiency
COMPILED_PATTERNS = {
    section: [re.compile(p, re.IGNORECASE) for p in patterns]
    for section, patterns in SECTION_PATTERNS.items()
}


def clean_html(html_text: str) -> str:
    """Strip HTML tags, normalize whitespace."""
    soup = BeautifulSoup(html_text, "lxml")

    # Remove scripts, styles, and hidden elements
    for tag in soup(["script", "style", "meta", "link", "header", "footer", "nav"]):
        tag.decompose()

    # Get text
    text = soup.get_text(separator="\n")

    # Normalize whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # Remove empty lines
    text = "\n".join(lines)

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text


def find_section_boundaries(text: str) -> dict[str, tuple[int, int]]:
    """
    Find start/end character positions for each target section.
    Returns dict: {section_name: (start_pos, end_pos)}
    """
    # Find all heading positions
    heading_hits = []  # (position, section_name)

    lines = text.split("\n")
    pos = 0

    for line in lines:
        line_stripped = line.strip()
        # Only check short lines likely to be headings
        if 2 < len(line_stripped) < 120:
            for section_name, patterns in COMPILED_PATTERNS.items():
                for pattern in patterns:
                    if pattern.search(line_stripped):
                        heading_hits.append((pos, section_name))
                        break

        pos += len(line) + 1  # +1 for newline

    # Deduplicate: keep first occurrence of each section
    seen = set()
    unique_hits = []
    for pos, name in sorted(heading_hits):
        if name not in seen:
            unique_hits.append((pos, name))
            seen.add(name)

    # Assign end positions
    boundaries = {}
    for i, (start_pos, name) in enumerate(unique_hits):
        # End is either start of next section or end of text
        if i + 1 < len(unique_hits):
            end_pos = unique_hits[i + 1][0]
        else:
            end_pos = len(text)
        boundaries[name] = (start_pos, end_pos)

    return boundaries


def extract_sections(html_path: Path) -> dict[str, str]:
    """
    Full pipeline: HTML → clean text → extract sections.
    Returns dict of {section_name: text}
    """
    try:
        raw = html_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"error": str(e)}

    text = clean_html(raw)
    boundaries = find_section_boundaries(text)

    sections = {"full_text_length": len(text)}

    for section_name, (start, end) in boundaries.items():
        section_text = text[start:end].strip()
        # Truncate to max budget
        if len(section_text) > MAX_SECTION_CHARS:
            section_text = section_text[:MAX_SECTION_CHARS] + "\n[TRUNCATED]"
        sections[section_name] = section_text

    # If no sections found, fall back to chunking full text
    found_sections = [k for k in sections if k not in ["full_text_length", "error"]]
    if not found_sections:
        sections["full_text_fallback"] = text[:MAX_SECTION_CHARS]

    return sections


def process_all_filings(manifest_path: Path) -> dict[str, dict]:
    """Process all filings in manifest, save section JSONs."""
    import pandas as pd
    manifest = pd.read_csv(manifest_path)
    ok = manifest[manifest["status"] == "ok"].copy()

    results = {}
    for _, row in ok.iterrows():
        ticker = row["ticker"]
        file_path = Path(row["file_path"])

        if not file_path.exists():
            print(f"  File not found: {file_path}")
            continue

        # One file per ticker — 424B4 preferred (manifest is sorted that way by edgar_fetcher)
        output_path = SECTIONS_DIR / f"{ticker}.json"

        if output_path.exists():
            print(f"  {ticker}: already processed — skipping")
            with open(output_path) as f:
                results[ticker] = json.load(f)
            continue

        print(f"  Extracting sections: {ticker} ({row['filing_type']})")
        sections = extract_sections(file_path)
        sections["ticker"] = ticker
        sections["filing_type"] = row["filing_type"]
        sections["filing_date"] = row["filing_date"]

        with open(output_path, "w") as f:
            json.dump(sections, f, indent=2)

        results[ticker] = sections
        found = [k for k in sections if k not in ["full_text_length", "ticker", "filing_type", "filing_date", "error"]]
        print(f"    Found sections: {found}")

    print(f"\nProcessed {len(results)} filings → {SECTIONS_DIR}")
    return results


if __name__ == "__main__":
    manifest_path = PROCESSED_DIR / "filing_manifest.csv"
    if manifest_path.exists():
        results = process_all_filings(manifest_path)
    else:
        print("No manifest found. Run edgar_fetcher.py first.")
