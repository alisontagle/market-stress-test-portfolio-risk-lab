"""Portfolio optimization and simulation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def annual_statistics(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    return returns.mean() * TRADING_DAYS, returns.cov() * TRADING_DAYS


def portfolio_statistics(
    weights: np.ndarray, mean_returns: pd.Series, covariance: pd.DataFrame, risk_free: float
) -> tuple[float, float, float]:
    expected_return = float(weights @ mean_returns.to_numpy())
    volatility = float(np.sqrt(weights @ covariance.to_numpy() @ weights))
    sharpe = (expected_return - risk_free) / volatility if volatility else 0.0
    return expected_return, volatility, sharpe


def optimize_portfolio(
    returns: pd.DataFrame, objective: str = "max_sharpe", risk_free: float = 0.02
) -> np.ndarray:
    mean_returns, covariance = annual_statistics(returns)
    number_assets = len(mean_returns)
    initial = np.repeat(1 / number_assets, number_assets)
    bounds = [(0.0, 1.0)] * number_assets
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def loss(weights: np.ndarray) -> float:
        _, volatility, sharpe = portfolio_statistics(weights, mean_returns, covariance, risk_free)
        return volatility if objective == "min_volatility" else -sharpe

    result = minimize(loss, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x


def simulate_portfolios(
    returns: pd.DataFrame, simulations: int = 5_000, risk_free: float = 0.02, seed: int = 44
) -> pd.DataFrame:
    mean_returns, covariance = annual_statistics(returns)
    rng = np.random.default_rng(seed)
    records: list[dict[str, float]] = []
    for _ in range(simulations):
        weights = rng.dirichlet(np.ones(len(mean_returns)))
        expected_return, volatility, sharpe = portfolio_statistics(
            weights, mean_returns, covariance, risk_free
        )
        record = {"Return": expected_return, "Volatility": volatility, "Sharpe": sharpe}
        record.update({f"Weight_{ticker}": weight for ticker, weight in zip(returns.columns, weights)})
        records.append(record)
    return pd.DataFrame(records)

