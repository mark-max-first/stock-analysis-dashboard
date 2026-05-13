import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("Stock Analysis Dashboard")
st.write(
    "Enter a stock ticker to analyze historical performance, risk, and benchmark comparison against SPY."
)

# Sidebar inputs
st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Enter stock ticker", value="AAPL").upper()
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (%)", value=4.0, step=0.25
) / 100

# Download 5 years of stock and SPY data
stock_data = yf.download(ticker, period="5y", auto_adjust=True)
spy_data = yf.download("SPY", period="5y", auto_adjust=True)

# Stop the app if the ticker is invalid
if stock_data.empty:
    st.error("No data found. Please enter a valid stock ticker.")
    st.stop()

# Use closing prices
# yfinance can sometimes return a one-column DataFrame instead of a Series, so squeeze() keeps it simple.
stock_prices = stock_data["Close"].squeeze()
spy_prices = spy_data["Close"].squeeze()

# Daily returns measure the day-to-day percentage change in price
stock_returns = stock_prices.pct_change().dropna()
spy_returns = spy_prices.pct_change().dropna()

# Annualized return estimates average yearly return from daily returns
annualized_return = float(stock_returns.mean() * 252)

# Annualized volatility measures yearly risk based on daily return movement
annualized_volatility = float(stock_returns.std() * np.sqrt(252))

# Sharpe ratio measures return earned per unit of risk
sharpe_ratio = float((annualized_return - risk_free_rate) / annualized_volatility)

# Beta measures how much the stock moves compared to SPY
combined_returns = pd.concat([stock_returns, spy_returns], axis=1).dropna()
combined_returns.columns = ["Stock", "SPY"]
beta = float(combined_returns["Stock"].cov(combined_returns["SPY"]) / combined_returns["SPY"].var())

# Max drawdown measures the largest drop from a previous high
running_max = stock_prices.cummax()
drawdown = (stock_prices / running_max) - 1
max_drawdown = float(drawdown.min())

# Most recent price
latest_price = float(stock_prices.iloc[-1])

# Show key metrics
st.subheader(f"{ticker} Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Most Recent Price", f"${latest_price:.2f}")
col2.metric("Annualized Return", f"{annualized_return:.2%}")
col3.metric("Annualized Volatility", f"{annualized_volatility:.2%}")

col4, col5, col6 = st.columns(3)
col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col5.metric("Beta vs SPY", f"{beta:.2f}")
col6.metric("Max Drawdown", f"{max_drawdown:.2%}")

# Normalize prices so both start at 100
normalized_stock = stock_prices / stock_prices.iloc[0] * 100
normalized_spy = spy_prices / spy_prices.iloc[0] * 100

st.subheader("Stock Performance vs SPY")

fig, ax = plt.subplots()
ax.plot(normalized_stock.index, normalized_stock, label=ticker)
ax.plot(normalized_spy.index, normalized_spy, label="SPY")
ax.set_title(f"{ticker} vs SPY Growth of $100")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.legend()
st.pyplot(fig)

st.subheader("Daily Returns")

fig2, ax2 = plt.subplots()
ax2.plot(stock_returns.index, stock_returns)
ax2.set_title(f"{ticker} Daily Returns")
ax2.set_xlabel("Date")
ax2.set_ylabel("Daily Return")
st.pyplot(fig2)

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