"""Input validation helpers that do not depend on Streamlit."""

import re


TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,10}$")


def clean_ticker(value):
    """Normalize a US equity or ETF ticker, or return an empty string."""
    ticker = value.strip().upper()
    return ticker if TICKER_PATTERN.fullmatch(ticker) else ""


def parse_tickers(ticker_text):
    """Return unique valid tickers and rejected entries from comma-separated text."""
    valid, invalid = [], []
    for raw_value in ticker_text.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        ticker = clean_ticker(raw_value)
        if not ticker:
            invalid.append(raw_value)
        elif ticker not in valid:
            valid.append(ticker)
    return valid, invalid
