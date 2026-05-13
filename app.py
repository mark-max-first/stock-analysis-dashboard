import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from datetime import date, timedelta

# Page setup
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("Stock Analysis Dashboard")

st.markdown(
    """
    Use this dashboard to review a stock's historical performance, risk profile,
    and benchmark comparison against SPY. Choose a ticker and date range in the
    sidebar, then review the key return and risk metrics below.
    """
)

st.subheader("How to Read This Dashboard")
st.write(
    "Start with the key metrics to understand return, risk, and market sensitivity. "
    "Then use the charts to compare performance against SPY and inspect daily price swings."
)
st.caption(
    "Note: Historical performance is for analysis only and does not guarantee future results."
)

# Sidebar inputs
st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Enter stock ticker", value="AAPL").strip().upper()
default_end_date = date.today()
default_start_date = default_end_date - timedelta(days=365 * 5)
start_date = st.sidebar.date_input("Start date", value=default_start_date)
end_date = st.sidebar.date_input("End date", value=default_end_date)
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (%)", value=4.0, step=0.25
) / 100

st.sidebar.subheader("Input Guide")
st.sidebar.write(
    """
    **Ticker:** Public stock symbol, such as AAPL or JPM.

    **Start and end dates:** The historical window used for prices, returns, and risk metrics.

    **Risk-free rate:** Annual rate used in the Sharpe ratio calculation.
    """
)

# Stop the app if the inputs are missing or the date range does not make sense
if not ticker:
    st.error("Please enter a stock ticker.")
    st.stop()

if start_date >= end_date:
    st.error("The start date must be before the end date.")
    st.stop()

# Download stock and SPY data using the selected dates
try:
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    spy_data = yf.download("SPY", start=start_date, end=end_date, auto_adjust=True)
except Exception as error:
    st.error(f"Could not download market data. Please try again. Error: {error}")
    st.stop()

# Stop the app if yfinance did not return usable data
if stock_data.empty:
    st.error(
        f"No price data found for {ticker}. Please check the ticker symbol or choose a wider date range."
    )
    st.stop()

if spy_data.empty:
    st.error("No benchmark data found for SPY. Please try a different date range.")
    st.stop()

if "Close" not in stock_data.columns or "Close" not in spy_data.columns:
    st.error("Downloaded data is missing closing prices. Please try again.")
    st.stop()

# Use closing prices. yfinance can sometimes return a one-column DataFrame,
# so squeeze() keeps the data simple to work with.
stock_prices = stock_data["Close"].squeeze().dropna()
spy_prices = spy_data["Close"].squeeze().dropna()

if len(stock_prices) < 2 or len(spy_prices) < 2:
    st.error("Not enough data found to calculate returns. Please choose a wider date range.")
    st.stop()

# Daily returns measure the day-to-day percentage change in price
stock_returns = stock_prices.pct_change().dropna()
spy_returns = spy_prices.pct_change().dropna()

if stock_returns.empty or spy_returns.empty:
    st.error("Not enough daily return data found. Please choose a wider date range.")
    st.stop()

# Annualized return estimates average yearly return from daily returns.
# There are about 252 trading days in a year, so daily average return is multiplied by 252.
annualized_return = float(stock_returns.mean() * 252)

# Annualized volatility measures yearly risk based on daily return movement.
# Daily standard deviation is multiplied by the square root of 252 trading days.
annualized_volatility = float(stock_returns.std() * np.sqrt(252))

# Sharpe ratio measures return earned per unit of risk.
# It compares stock return above the risk-free rate to the stock's volatility.
if annualized_volatility == 0:
    sharpe_ratio = np.nan
else:
    sharpe_ratio = float((annualized_return - risk_free_rate) / annualized_volatility)

# Beta measures how much the stock moves compared to SPY.
# It is the covariance of stock and market returns divided by market variance.
combined_returns = pd.concat([stock_returns, spy_returns], axis=1).dropna()
combined_returns.columns = ["Stock", "SPY"]
spy_variance = combined_returns["SPY"].var()
if combined_returns.empty or spy_variance == 0:
    beta = np.nan
else:
    beta = float(combined_returns["Stock"].cov(combined_returns["SPY"]) / spy_variance)

# Max drawdown measures the largest drop from a previous high.
# It compares each price with the highest price reached before that date.
running_max = stock_prices.cummax()
drawdown = (stock_prices / running_max) - 1
max_drawdown = float(drawdown.min())

# Most recent price
latest_price = float(stock_prices.iloc[-1])

# Show key metrics
st.divider()
st.subheader(f"{ticker} Key Metrics")
st.caption(
    f"Analysis period: {stock_prices.index.min().date()} to {stock_prices.index.max().date()}"
)

col1, col2, col3 = st.columns(3)
col1.metric("Most Recent Price", f"${latest_price:.2f}")
col2.metric("Annualized Return", f"{annualized_return:.2%}")
col3.metric("Annualized Volatility", f"{annualized_volatility:.2%}")

col4, col5, col6 = st.columns(3)
col4.metric("Sharpe Ratio", "N/A" if np.isnan(sharpe_ratio) else f"{sharpe_ratio:.2f}")
col5.metric("Beta vs SPY", "N/A" if np.isnan(beta) else f"{beta:.2f}")
col6.metric("Max Drawdown", f"{max_drawdown:.2%}")

# Normalize prices so both start at 100
normalized_stock = stock_prices / stock_prices.iloc[0] * 100
normalized_spy = spy_prices / spy_prices.iloc[0] * 100

st.subheader("Stock Performance vs SPY")
st.write(
    "This chart compares the growth of a hypothetical $100 investment in the selected stock and SPY."
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(normalized_stock.index, normalized_stock, label=ticker)
ax.plot(normalized_spy.index, normalized_spy, label="SPY")
ax.set_title(f"{ticker} vs SPY Growth of $100")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
st.pyplot(fig, use_container_width=True)

st.subheader("Daily Returns")
st.write(
    "Daily returns show how much the stock moved up or down each trading day. Larger swings usually mean higher risk."
)

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(stock_returns.index, stock_returns)
ax2.set_title(f"{ticker} Daily Returns")
ax2.set_xlabel("Date")
ax2.set_ylabel("Daily Return")
ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
st.pyplot(fig2, use_container_width=True)

# Downloadable data
st.subheader("Download Data")
download_data = stock_data.copy()
download_data["Daily Return"] = stock_returns

csv = download_data.to_csv().encode("utf-8")

st.download_button(
    label="Download Stock Data as CSV",
    data=csv,
    file_name=f"{ticker}_stock_data.csv",
    mime="text/csv",
)

# Simple explanations
st.subheader("What These Metrics Mean")

st.write("""
**Annualized Return:** The estimated yearly return based on average daily returns.

**Annualized Volatility:** A measure of risk. Higher volatility means the stock price moves around more.

**Sharpe Ratio:** Shows how much return the stock generated for each unit of risk. Higher is usually better.

**Beta vs SPY:** Measures how sensitive the stock is compared to the overall market. A beta above 1 means the stock is more volatile than SPY.

**Max Drawdown:** The largest percentage drop from a previous high. This helps show downside risk.
""")
