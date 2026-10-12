# EquityLens: Stock Analysis Dashboard

[![Tests](https://github.com/mark-max-first/stock-analysis-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/mark-max-first/stock-analysis-dashboard/actions/workflows/tests.yml)

[Open the live dashboard](https://stock-analysis-dashboard-mark.streamlit.app) · [View Mark Spinelli's GitHub profile](https://github.com/mark-max-first)

EquityLens is an interactive Python and Streamlit project for evaluating a US-listed equity or ETF's historical return, volatility, downside risk, and sensitivity to the S&P 500 ETF (SPY). It also includes a transparent multi-stock screener for comparing up to 12 tickers over a selected period using each ticker's available history.

This is an educational analytics project, not investment advice or a trading system.

## What the dashboard does

- Analyzes supported US-listed equity and ETF tickers over a custom date range
- Compares the growth of $100 in the selected stock with SPY
- Calculates total return, compound annual growth rate (CAGR), annualized volatility, Sharpe ratio, beta, and maximum drawdown
- Visualizes adjusted prices, 50-day and 200-day moving averages, daily returns, return distribution, and drawdowns
- Screens and ranks multiple stocks with a documented weighted-percentile model
- Exports both single-stock analysis data and screener results to CSV
- Explains the methodology and limitations inside the app

## Why I built it

I built this project to connect finance coursework with practical data analysis. The goal was to turn raw historical prices into a clear, interactive review of return, risk, benchmark sensitivity, and downside behavior.

The project demonstrates:

- Translating finance concepts into tested calculations
- Working with market data in pandas
- Designing an interactive analytical workflow in Streamlit
- Communicating model assumptions and limitations clearly
- Separating calculation logic from the user interface so it can be tested independently

## Technology

- Python
- Streamlit
- pandas and NumPy
- Matplotlib
- yfinance
- Python `unittest`

## Calculation methodology

| Metric | Method |
|---|---|
| Total return | Last adjusted close divided by first adjusted close, minus one |
| CAGR | Compounded change in adjusted close over elapsed calendar time; shown for periods of at least one year |
| Annualized volatility | Daily-return standard deviation multiplied by √252 |
| Sharpe ratio | Annualized average daily excess return divided by daily volatility |
| Beta vs SPY | Covariance of aligned stock and SPY daily returns divided by SPY variance |
| Maximum drawdown | Largest percentage decline from a previous adjusted-price peak |

The screener score weights Sharpe ratio 35%, CAGR 30%, maximum drawdown 20%, and annualized volatility 15%. It ranks the submitted list only, so scores cannot be compared across separate runs. A missing Sharpe ratio receives no contribution from that component. It is descriptive, not predictive.

## Run locally

```bash
git clone https://github.com/mark-max-first/stock-analysis-dashboard.git
cd stock-analysis-dashboard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Run the tests

```bash
python -m unittest discover -s tests -v
```

The automated checks cover total return, CAGR, volatility, Sharpe ratio, beta alignment, maximum drawdown, short-history and flat-price edge cases, yfinance data shapes, screener ranking, and ticker validation.

## Repository structure

```text
app.py                 Streamlit interface and market-data workflow
finance.py             Testable finance calculations and ranking logic
validation.py          Ticker normalization and validation
tests/                 Automated calculation and input checks
requirements.txt       Direct Python dependencies
```

## Data and limitations

Market prices are downloaded through `yfinance` and adjusted for corporate actions by the provider. Availability and accuracy depend on the upstream service. Historical results change with the chosen date range and do not predict future performance. The risk-free rate is a user-selected assumption.

## Next extensions

- Add company fundamentals and valuation multiples from a documented data source
- Add portfolio-level allocation and correlation analysis
- Compare model outputs with a second market-data source

## License

MIT License. See [LICENSE](LICENSE).
