"""Interactive Market Stress-Test & Portfolio Risk Lab."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.backtest import wealth_index
from src.data import calculate_returns, download_prices
from src.portfolio import optimize_portfolio, simulate_portfolios
from src.risk import historical_stress, hypothetical_shock, portfolio_returns, risk_metrics

st.set_page_config(page_title="Market Risk Lab", page_icon="📈", layout="wide")
st.title("Market Stress-Test & Portfolio Risk Lab")
st.caption("Portfolio optimization, Monte Carlo simulation, backtesting, and downside-risk analysis")

ASSETS = {
    "S&P 500 (SPY)": "SPY", "Nasdaq 100 (QQQ)": "QQQ", "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT", "JPMorgan (JPM)": "JPM", "Gold (GLD)": "GLD",
    "Treasury Bonds (IEF)": "IEF", "Real Estate (VNQ)": "VNQ",
}

with st.sidebar:
    st.header("Research settings")
    selected_names = st.multiselect("Assets", list(ASSETS), default=list(ASSETS)[:6])
    start = st.date_input("Start date", date(2018, 1, 1))
    end = st.date_input("End date", date.today())
    risk_free = st.slider("Risk-free rate", 0.0, 0.10, 0.03, 0.005)
    simulations = st.slider("Monte Carlo portfolios", 1_000, 10_000, 5_000, 1_000)
    run = st.button("Run analysis", type="primary", use_container_width=True)

if not run:
    st.info("Choose the assets and dates, then click **Run analysis**.")
    st.markdown("""
    **Research question:** Can quantitative portfolio construction improve risk-adjusted performance
    and downside protection relative to a simple equal-weighted strategy?

    This dashboard compares equal-weighted, minimum-volatility, and maximum-Sharpe portfolios.
    It also evaluates historical crashes and hypothetical asset-class shocks.
    """)
    st.stop()

tickers = [ASSETS[name] for name in selected_names]
if len(tickers) < 2:
    st.error("Please select at least two assets.")
    st.stop()

try:
    with st.spinner("Downloading and analyzing market data..."):
        prices = download_prices(tickers, str(start), str(end))
        returns = calculate_returns(prices)
        equal_weights = np.repeat(1 / len(returns.columns), len(returns.columns))
        min_vol_weights = optimize_portfolio(returns, "min_volatility", risk_free)
        max_sharpe_weights = optimize_portfolio(returns, "max_sharpe", risk_free)
        portfolios = {
            "Equal Weight": portfolio_returns(returns, equal_weights),
            "Minimum Volatility": portfolio_returns(returns, min_vol_weights),
            "Maximum Sharpe": portfolio_returns(returns, max_sharpe_weights),
        }
except Exception as exc:
    st.error(f"Analysis could not run: {exc}")
    st.stop()

portfolio_table = pd.DataFrame(portfolios)
metric_table = pd.DataFrame({name: risk_metrics(series, risk_free=risk_free) for name, series in portfolios.items()}).T

tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Optimization", "Risk", "Stress tests"])

with tab1:
    st.subheader("Growth of a $10,000 investment")
    wealth = wealth_index(portfolio_table)
    st.plotly_chart(px.line(wealth, labels={"value": "Portfolio value ($)", "Date": "Date", "variable": "Strategy"}), use_container_width=True)
    formatted = metric_table.copy()
    for column in ["Annual Return", "Annual Volatility", "Maximum Drawdown", "95% Daily VaR", "95% Daily CVaR", "Downside Volatility"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")
    formatted["Sharpe Ratio"] = formatted["Sharpe Ratio"].map(lambda value: f"{value:.2f}")
    st.dataframe(formatted, use_container_width=True)

with tab2:
    st.subheader("Monte Carlo opportunity set")
    simulation = simulate_portfolios(returns, simulations, risk_free)
    figure = px.scatter(simulation, x="Volatility", y="Return", color="Sharpe", color_continuous_scale="Viridis", opacity=0.55)
    figure.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(figure, use_container_width=True)
    weights = pd.DataFrame({
        "Asset": returns.columns,
        "Equal Weight": equal_weights,
        "Minimum Volatility": min_vol_weights,
        "Maximum Sharpe": max_sharpe_weights,
    }).set_index("Asset")
    st.plotly_chart(px.bar(weights, barmode="group", labels={"value": "Portfolio weight", "variable": "Strategy"}), use_container_width=True)

with tab3:
    st.subheader("Asset correlations")
    heatmap = go.Figure(go.Heatmap(z=returns.corr(), x=returns.columns, y=returns.columns, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True))
    st.plotly_chart(heatmap, use_container_width=True)
    st.caption("VaR estimates the daily loss threshold at a selected confidence level. CVaR estimates the average loss beyond that threshold.")

with tab4:
    st.subheader("Historical stress periods")
    periods = {"COVID crash": ("2020-02-19", "2020-03-23"), "2022 rate shock": ("2022-01-03", "2022-10-12")}
    rows = []
    for event, (event_start, event_end) in periods.items():
        for strategy, series in portfolios.items():
            rows.append({"Event": event, "Strategy": strategy, **historical_stress(series, event_start, event_end)})
    stress = pd.DataFrame(rows)
    for column in ["Cumulative Return", "Worst Day", "Maximum Drawdown"]:
        stress[column] = stress[column].map(lambda value: "N/A" if pd.isna(value) else f"{value:.2%}")
    st.dataframe(stress, use_container_width=True)

    st.subheader("Hypothetical cross-asset shock")
    default_shocks = {"SPY": -0.20, "QQQ": -0.25, "AAPL": -0.25, "MSFT": -0.22, "JPM": -0.18, "GLD": 0.08, "IEF": 0.04, "VNQ": -0.20}
    shocks = np.array([default_shocks.get(ticker, -0.15) for ticker in returns.columns])
    shock_results = pd.Series({
        "Equal Weight": hypothetical_shock(equal_weights, shocks),
        "Minimum Volatility": hypothetical_shock(min_vol_weights, shocks),
        "Maximum Sharpe": hypothetical_shock(max_sharpe_weights, shocks),
    }, name="Estimated one-period loss")
    st.dataframe(shock_results.map(lambda value: f"{value:.2%}").to_frame(), use_container_width=True)
    st.caption("Scenario assumptions: equities and real estate decline while gold and intermediate Treasuries rise. This is a sensitivity analysis, not a forecast.")

st.divider()
st.caption("Educational research project. Historical results do not guarantee future performance and are not investment advice.")

