"""Simple portfolio backtesting helpers."""

from __future__ import annotations

import pandas as pd


def wealth_index(returns: pd.DataFrame | pd.Series, initial: float = 10_000) -> pd.DataFrame | pd.Series:
    """Growth of an initial investment, with the first observed day as the base."""
    return initial * (1 + returns).cumprod()

