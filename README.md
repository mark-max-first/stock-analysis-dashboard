# Stock Analysis Dashboard

A beginner-friendly finance dashboard built with Python and Streamlit. The app lets a user enter a stock ticker and analyze historical stock performance, risk, and benchmark performance against SPY.

## Project Overview

This project was built as a finance internship resume project. It uses market data to calculate key investment metrics that analysts use to evaluate risk and return.

The dashboard also visualizes downside risk over time with a drawdown chart that shows how far a stock fell from previous highs.

It also includes basic trend analysis using 50-day and 200-day moving averages.

The dashboard helps users understand volatility using both annualized volatility and the distribution of daily returns.

The dashboard helps answer questions like:

- How has a stock performed over the last 5 years?
- How does the stock compare to the S&P 500 ETF, SPY?
- How risky is the stock based on volatility and drawdown?
- How much return did the stock generate compared to its risk?

## Features

- Enter any public stock ticker
- Pull 5 years of historical stock data using yfinance
- Compare stock performance against SPY
- Calculate daily returns
- Calculate annualized return
- Calculate annualized volatility
- Calculate Sharpe ratio
- Calculate beta vs SPY
- Calculate max drawdown
- Visualize drawdown over time with a Drawdown Analysis chart
- Plot 50-day and 200-day moving averages for basic trend analysis
- View a returns distribution chart showing the spread of daily returns
- Show most recent stock price
- Download stock data as a CSV file
- View simple explanations of each finance metric

## Tech Stack

- Python
- Streamlit
- pandas
- numpy
- matplotlib
- yfinance

## Finance Metrics Explained

### Annualized Return

Annualized return estimates the average yearly return of the stock based on historical daily returns.

### Annualized Volatility

Annualized volatility measures how much the stock price moves around over time. Higher volatility means higher risk.

### Sharpe Ratio

The Sharpe ratio measures return per unit of risk. A higher Sharpe ratio usually means the stock generated better risk-adjusted returns.

### Beta vs SPY

Beta measures how sensitive the stock is compared to SPY, which is used as a market benchmark. A beta above 1 means the stock is more volatile than the market.

### Max Drawdown

Max drawdown shows the largest percentage drop from a previous high. The Drawdown Analysis chart visualizes this downside risk over time, making it easier to see when the stock experienced its largest declines.

### Moving Averages

Moving averages smooth daily price data by averaging the stock's closing price over a set number of trading days. The 50-day moving average helps show shorter-term trends, while the 200-day moving average helps show longer-term trends.

### Returns Distribution

The returns distribution chart shows how daily returns are spread out over time. A wider distribution usually means the stock had higher volatility, while a narrower distribution usually means daily returns were more stable.

## How to Run the App

1. Clone or download this project.

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## Example Tickers to Try

- AAPL
- MSFT
- JPM
- NVDA
- TSLA
- SPY

## What I Learned

Through this project, I practiced using Python to pull financial market data, calculate investment risk metrics, compare a stock against a benchmark, and build an interactive dashboard that presents results clearly.

I also used Codex to help build, debug, and improve the project while reviewing the finance logic myself.

## Resume Bullets

- Built a Python-based stock analysis dashboard using Streamlit, pandas, and yfinance to analyze public equities and compare performance against SPY.
- Calculated investment metrics including annualized return, annualized volatility, Sharpe ratio, beta, and max drawdown to evaluate stock risk and return.
- Created a clean interactive dashboard with downloadable data and simple explanations of key finance metrics.

## Future Improvements

- Add company financial statement data
- Add valuation ratios like P/E, EV/EBITDA, and dividend yield
- Add moving averages and technical indicators
- Add portfolio tracking for multiple stocks
- Add AI-generated stock summaries
- Deploy the dashboard online using Streamlit Community Cloud
