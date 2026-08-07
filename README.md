# Market Stress-Test & Portfolio Risk Lab

An interactive quantitative-finance project by **Alison Tagle**. The project studies whether
portfolio optimization can improve risk-adjusted performance and downside protection relative
to a simple equal-weighted strategy.

## What the project demonstrates

- Python data collection and cleaning
- Return, volatility, correlation, Sharpe ratio, and drawdown analysis
- Maximum-Sharpe and minimum-volatility optimization
- Monte Carlo simulation of thousands of possible portfolios
- Value at Risk (VaR) and Conditional Value at Risk (CVaR)
- Historical and hypothetical stress testing
- Interactive financial-data visualization

## Run it on a Mac or Windows computer

1. Install Python 3.11 or newer from <https://www.python.org/downloads/>.
2. Open Terminal (Mac) or Command Prompt (Windows).
3. Move into this project folder. Example:

   ```bash
   cd Downloads/market-risk-lab
   ```

4. Create a private environment:

   ```bash
   python3 -m venv .venv
   ```

5. Activate it on Mac:

   ```bash
   source .venv/bin/activate
   ```

   On Windows:

   ```powershell
   .venv\Scripts\activate
   ```

6. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

7. Launch the dashboard:

   ```bash
   streamlit run app.py
   ```

## Methodology

Daily adjusted closing prices are converted into percentage returns. Expected returns and the
covariance matrix are annualized using 252 trading days. Long-only portfolios are fully invested,
so weights are between 0% and 100% and sum to 100%. Numerical optimization identifies the
minimum-volatility and maximum-Sharpe portfolios. Results are compared with an equal-weighted
benchmark using historical backtesting and downside-risk measures.

## Important limitations

- Optimization uses historical estimates, which may be unstable out of sample.
- The initial version does not model transaction costs, taxes, or liquidity.
- VaR and CVaR are historical estimates and depend on the selected sample.
- Hypothetical stress tests are sensitivity analyses, not forecasts.

## Recommended next research extensions

1. Add rolling, out-of-sample portfolio rebalancing.
2. Include transaction costs and turnover constraints.
3. Compare covariance shrinkage with the sample covariance matrix.
4. Identify market regimes with a hidden Markov model.
5. Test whether the optimized strategies remain superior after costs.

## Author

**Alison Tagle**  
Finance student at San Diego State University with interests in portfolio management, financial risk analysis, quantitative finance, and wealth management.

