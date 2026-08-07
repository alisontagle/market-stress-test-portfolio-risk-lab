"""Risk measurement, drawdown, and scenario testing."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return returns @ weights


def maximum_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return float(drawdown.min())


def risk_metrics(returns: pd.Series, confidence: float = 0.95, risk_free: float = 0.02) -> dict[str, float]:
    annual_return = float(returns.mean() * TRADING_DAYS)
    annual_volatility = float(returns.std() * np.sqrt(TRADING_DAYS))
    cutoff = float(returns.quantile(1 - confidence))
    tail = returns[returns <= cutoff]
    cvar = float(tail.mean()) if not tail.empty else cutoff
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    return {
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": (annual_return - risk_free) / annual_volatility if annual_volatility else 0.0,
        "Maximum Drawdown": maximum_drawdown(returns),
        f"{confidence:.0%} Daily VaR": cutoff,
        f"{confidence:.0%} Daily CVaR": cvar,
        "Downside Volatility": float(downside),
    }


def historical_stress(returns: pd.Series, start: str, end: str) -> dict[str, float]:
    period = returns.loc[start:end]
    if period.empty:
        return {"Cumulative Return": np.nan, "Worst Day": np.nan, "Maximum Drawdown": np.nan}
    return {
        "Cumulative Return": float((1 + period).prod() - 1),
        "Worst Day": float(period.min()),
        "Maximum Drawdown": maximum_drawdown(period),
    }


def hypothetical_shock(weights: np.ndarray, shocks: np.ndarray) -> float:
    return float(weights @ shocks)

