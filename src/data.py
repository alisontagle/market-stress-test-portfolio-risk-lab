"""Market-data loading and return calculations."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted daily closing prices and return a clean table."""
    if not tickers:
        raise ValueError("Select at least one asset.")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("No market data were returned. Check the dates and internet connection.")
    prices = raw["Close"] if "Close" in raw.columns else raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.dropna(axis=1, how="all").ffill().dropna()
    if prices.empty:
        raise ValueError("The selected assets have no overlapping price history.")
    return prices


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert prices to simple daily percentage returns."""
    return prices.pct_change(fill_method=None).dropna()

