import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Create dirs if missing
for d in [RAW_DIR, PROCESSED_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# EDGAR
EDGAR_USER_AGENT = "Robert_Lennon robertlennon2021@gmail.com"  # Required by SEC

# IPO Universe
# Date range for IPO sample — adjust as needed
IPO_START_YEAR = 2015
IPO_END_YEAR = 2023
MIN_OFFER_SIZE_M = 50  # Filter tiny IPOs (< $50M raised)

# Target windows (trading days)
RETURN_WINDOWS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "6m": 126,
    "1y": 252,
}

# Modeling
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Sections to extract from filings
TARGET_SECTIONS = [
    "summary",
    "risk factors",
    "business",
    "use of proceeds",
    "management",
]

# Uncertainty / hedging language keywords
UNCERTAINTY_KEYWORDS = [
    "may", "might", "could", "would", "should", "expect", "anticipate",
    "intend", "plan", "believe", "estimate", "potential", "possible",
    "uncertainty", "risk", "uncertain", "no assurance", "cannot guarantee",
]

# Growth framing keywords
GROWTH_KEYWORDS = [
    "growth", "expand", "increase", "accelerate", "opportunity", "market",
    "scale", "revenue", "customers", "acquisition", "platform",
]

# Profitability keywords
PROFIT_KEYWORDS = [
    "profit", "profitable", "profitability", "net income", "earnings",
    "EBITDA", "margin", "cash flow",
]

LOSS_KEYWORDS = [
    "loss", "deficit", "negative", "accumulated deficit", "net loss",
    "cash burn", "not profitable", "never been profitable",
]
