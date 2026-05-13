from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import PercentFormatter


# Page setup
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")


def download_stock_data(ticker, start_date, end_date):
    """Download adjusted price data for one ticker."""
    return yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)


def get_close_prices(price_data):
    """Pull the Close column from yfinance data and return it as a simple Series."""
    if price_data.empty:
        return pd.Series(dtype=float)

    if isinstance(price_data.columns, pd.MultiIndex):
        if "Close" not in price_data.columns.get_level_values(0):
            return pd.Series(dtype=float)
        close_prices = price_data["Close"]
    else:
        if "Close" not in price_data.columns:
            return pd.Series(dtype=float)
        close_prices = price_data["Close"]

    return close_prices.squeeze().dropna()


def calculate_finance_metrics(stock_prices, spy_prices, risk_free_rate):
    """Calculate the main return and risk metrics used throughout the app."""
    if len(stock_prices) < 2 or len(spy_prices) < 2:
        return None

    # Daily returns measure the day-to-day percentage change in price.
    stock_returns = stock_prices.pct_change().dropna()
    spy_returns = spy_prices.pct_change().dropna()

    if stock_returns.empty or spy_returns.empty:
        return None

    # Annualized return estimates average yearly return from daily returns.
    # There are about 252 trading days in a year, so daily average return is multiplied by 252.
    annualized_return = float(stock_returns.mean() * 252)

    # Annualized volatility measures yearly risk based on daily return movement.
    # Daily standard deviation is multiplied by the square root of 252 trading days.
    annualized_volatility = float(stock_returns.std() * np.sqrt(252))

    # Sharpe ratio measures return earned per unit of risk.
    # It compares stock return above the risk-free rate to the stock's volatility.
    if annualized_volatility == 0 or np.isnan(annualized_volatility):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = float((annualized_return - risk_free_rate) / annualized_volatility)

    # Beta measures how much the stock moves compared to SPY.
    # It is the covariance of stock and market returns divided by market variance.
    combined_returns = pd.concat([stock_returns, spy_returns], axis=1).dropna()
    combined_returns.columns = ["Stock", "SPY"]
    spy_variance = combined_returns["SPY"].var()
    if combined_returns.empty or spy_variance == 0 or np.isnan(spy_variance):
        beta = np.nan
    else:
        beta = float(combined_returns["Stock"].cov(combined_returns["SPY"]) / spy_variance)

    # Max drawdown measures the largest drop from a previous high.
    # It compares each price with the highest price reached before that date.
    running_max = stock_prices.cummax()
    drawdown = (stock_prices / running_max) - 1
    max_drawdown = float(drawdown.min())

    return {
        "stock_returns": stock_returns,
        "spy_returns": spy_returns,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "beta": beta,
        "drawdown": drawdown,
        "max_drawdown": max_drawdown,
    }


def assign_category(row):
    """Give each stock a simple label based on its historical risk and return profile."""
    if row["Max Drawdown"] <= -0.35:
        return "Larger drawdown risk"
    if row["Sharpe Ratio"] >= 1 and row["Annualized Return"] > 0:
        return "Strong historical risk-adjusted return"
    if row["Annualized Return"] >= 0.15 and row["Annualized Volatility"] >= 0.25:
        return "Higher risk / higher return"
    if row["Annualized Volatility"] <= 0.20:
        return "Lower volatility profile"
    return "Mixed historical profile"


def add_ranking_score(results):
    """Create a simple score using return, risk-adjusted return, drawdown, and volatility."""
    scored_results = results.copy()

    # Higher Sharpe ratio and higher return should improve the score.
    sharpe_score = scored_results["Sharpe Ratio"].rank(pct=True).fillna(0)
    return_score = scored_results["Annualized Return"].rank(pct=True).fillna(0)

    # Max drawdown is usually negative, so values closer to 0 are better.
    drawdown_score = scored_results["Max Drawdown"].rank(pct=True).fillna(0)

    # Lower volatility is better for this score.
    volatility_score = scored_results["Annualized Volatility"].rank(
        pct=True, ascending=False
    ).fillna(0)

    scored_results["Ranking Score"] = (
        sharpe_score * 35
        + return_score * 30
        + drawdown_score * 20
        + volatility_score * 15
    )

    return scored_results.sort_values("Ranking Score", ascending=False)


def format_screener_table(results):
    """Format numeric columns so the table is easier to read."""
    display_results = results.copy()
    format_percent = lambda value: "N/A" if pd.isna(value) else f"{value:.2%}"
    format_number = lambda value: "N/A" if pd.isna(value) else f"{value:.2f}"

    display_results["Annualized Return"] = display_results["Annualized Return"].map(format_percent)
    display_results["Annualized Volatility"] = display_results["Annualized Volatility"].map(
        format_percent
    )
    display_results["Sharpe Ratio"] = display_results["Sharpe Ratio"].map(format_number)
    display_results["Beta vs SPY"] = display_results["Beta vs SPY"].map(format_number)
    display_results["Max Drawdown"] = display_results["Max Drawdown"].map(format_percent)
    display_results["Ranking Score"] = display_results["Ranking Score"].map(
        lambda value: "N/A" if pd.isna(value) else f"{value:.1f}"
    )
    return display_results


def render_single_stock_page(ticker, start_date, end_date, risk_free_rate):
    """Render the original single-stock dashboard."""
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

    if not ticker:
        st.error("Please enter a stock ticker.")
        st.stop()

    try:
        stock_data = download_stock_data(ticker, start_date, end_date)
        spy_data = download_stock_data("SPY", start_date, end_date)
    except Exception as error:
        st.error(f"Could not download market data. Please try again. Error: {error}")
        st.stop()

    if stock_data.empty:
        st.error(
            f"No price data found for {ticker}. Please check the ticker symbol or choose a wider date range."
        )
        st.stop()

    if spy_data.empty:
        st.error("No benchmark data found for SPY. Please try a different date range.")
        st.stop()

    stock_prices = get_close_prices(stock_data)
    spy_prices = get_close_prices(spy_data)

    if stock_prices.empty or spy_prices.empty:
        st.error("Downloaded data is missing closing prices. Please try again.")
        st.stop()

    metrics = calculate_finance_metrics(stock_prices, spy_prices, risk_free_rate)
    if metrics is None:
        st.error("Not enough data found to calculate returns. Please choose a wider date range.")
        st.stop()

    stock_returns = metrics["stock_returns"]
    annualized_return = metrics["annualized_return"]
    annualized_volatility = metrics["annualized_volatility"]
    sharpe_ratio = metrics["sharpe_ratio"]
    beta = metrics["beta"]
    drawdown = metrics["drawdown"]
    max_drawdown = metrics["max_drawdown"]

    # Moving averages smooth the stock price by averaging the last 50 or 200 trading days.
    moving_average_50 = stock_prices.rolling(window=50).mean()
    moving_average_200 = stock_prices.rolling(window=200).mean()

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

    # Normalize prices so both start at 100.
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

    st.subheader("Moving Average Analysis")
    st.write(
        "Moving averages smooth price data and help identify trends. The 50-day average reacts faster, while the 200-day average shows a longer-term trend."
    )

    fig_ma, ax_ma = plt.subplots(figsize=(12, 5))
    ax_ma.plot(stock_prices.index, stock_prices, label=f"{ticker} Close")
    ax_ma.plot(moving_average_50.index, moving_average_50, label="50-Day Moving Average")
    ax_ma.plot(moving_average_200.index, moving_average_200, label="200-Day Moving Average")
    ax_ma.set_title(f"{ticker} Price with Moving Averages")
    ax_ma.set_xlabel("Date")
    ax_ma.set_ylabel("Price")
    ax_ma.grid(True, alpha=0.3)
    ax_ma.legend()
    fig_ma.tight_layout()
    st.pyplot(fig_ma, use_container_width=True)

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

    st.subheader("Returns Distribution")
    st.write(
        "This chart shows how daily returns are distributed over time. Wider distributions usually mean higher volatility."
    )

    # A histogram groups daily returns into ranges so it is easier to see common outcomes.
    fig_dist, ax_dist = plt.subplots(figsize=(12, 5))
    ax_dist.hist(stock_returns, bins=40)
    ax_dist.set_title(f"{ticker} Daily Returns Distribution")
    ax_dist.set_xlabel("Daily Return")
    ax_dist.set_ylabel("Number of Trading Days")
    ax_dist.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax_dist.grid(True, alpha=0.3)
    fig_dist.tight_layout()
    st.pyplot(fig_dist, use_container_width=True)

    st.subheader("Drawdown Analysis")
    st.write(
        "Drawdown shows how far the stock fell from its previous high. Deeper drawdowns show larger downside moves."
    )

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(drawdown.index, drawdown)
    ax3.set_title(f"{ticker} Drawdown Over Time")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Drawdown")
    ax3.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)

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

    st.write(
        """
    **Annualized Return:** The estimated yearly return based on average daily returns.

    **Annualized Volatility:** A measure of risk. Higher volatility means the stock price moves around more.

    **Sharpe Ratio:** Shows how much return the stock generated for each unit of risk. Higher is usually better.

    **Beta vs SPY:** Measures how sensitive the stock is compared to the overall market. A beta above 1 means the stock is more volatile than SPY.

    **Max Drawdown:** The largest percentage drop from a previous high. This helps show downside risk.
    """
    )


def render_screener_page(ticker_text, start_date, end_date, risk_free_rate):
    """Render a simple stock screener table."""
    st.title("Stock Screener")

    st.markdown(
        """
        Compare multiple stocks over the same date range using return, risk,
        drawdown, beta, and Sharpe ratio. The ranking score is a simple educational
        model based on historical data.
        """
    )

    st.info(
        "This screener is for educational analysis only and is not investment advice."
    )

    tickers = []
    for ticker in ticker_text.split(","):
        clean_ticker = ticker.strip().upper()
        if clean_ticker and clean_ticker not in tickers:
            tickers.append(clean_ticker)

    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    try:
        spy_data = download_stock_data("SPY", start_date, end_date)
    except Exception as error:
        st.error(f"Could not download SPY benchmark data. Please try again. Error: {error}")
        st.stop()

    spy_prices = get_close_prices(spy_data)
    if spy_prices.empty:
        st.error("No benchmark data found for SPY. Please try a different date range.")
        st.stop()

    screener_rows = []
    skipped_tickers = []

    for ticker in tickers:
        try:
            stock_data = download_stock_data(ticker, start_date, end_date)
            stock_prices = get_close_prices(stock_data)
        except Exception:
            skipped_tickers.append(ticker)
            continue

        if stock_prices.empty:
            skipped_tickers.append(ticker)
            continue

        metrics = calculate_finance_metrics(stock_prices, spy_prices, risk_free_rate)
        if metrics is None:
            skipped_tickers.append(ticker)
            continue

        screener_rows.append(
            {
                "Ticker": ticker,
                "Annualized Return": metrics["annualized_return"],
                "Annualized Volatility": metrics["annualized_volatility"],
                "Sharpe Ratio": metrics["sharpe_ratio"],
                "Beta vs SPY": metrics["beta"],
                "Max Drawdown": metrics["max_drawdown"],
            }
        )

    if not screener_rows:
        st.error("No usable stock data found. Please check the tickers or choose a wider date range.")
        st.stop()

    results = pd.DataFrame(screener_rows)
    results = add_ranking_score(results)
    results["Category"] = results.apply(assign_category, axis=1)

    st.subheader("Screener Results")
    st.caption(f"Analysis period: {start_date} to {end_date}")

    display_columns = [
        "Ticker",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Beta vs SPY",
        "Max Drawdown",
        "Ranking Score",
        "Category",
    ]
    st.dataframe(
        format_screener_table(results[display_columns]),
        use_container_width=True,
        hide_index=True,
    )

    st.write(
        """
        **Ranking Score:** Higher scores favor stronger Sharpe ratios, higher annualized returns,
        smaller drawdowns, and lower volatility. This is a simple historical screen, not a prediction.
        """
    )

    if skipped_tickers:
        st.warning(
            "Skipped tickers with missing or incomplete data: " + ", ".join(skipped_tickers)
        )


# Sidebar page controls
page = st.sidebar.selectbox(
    "Select page",
    ["Single Stock Analysis", "Stock Screener"],
)

st.sidebar.header("Inputs")
default_end_date = date.today()
default_start_date = default_end_date - timedelta(days=365 * 5)

start_date = st.sidebar.date_input("Start date", value=default_start_date)
end_date = st.sidebar.date_input("End date", value=default_end_date)
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (%)", value=4.0, step=0.25
) / 100

if start_date >= end_date:
    st.error("The start date must be before the end date.")
    st.stop()

if page == "Single Stock Analysis":
    ticker = st.sidebar.text_input("Enter stock ticker", value="AAPL").strip().upper()

    st.sidebar.subheader("Input Guide")
    st.sidebar.write(
        """
        **Ticker:** Public stock symbol, such as AAPL or JPM.

        **Start and end dates:** The historical window used for prices, returns, and risk metrics.

        **Risk-free rate:** Annual rate used in the Sharpe ratio calculation.
        """
    )

    render_single_stock_page(ticker, start_date, end_date, risk_free_rate)

else:
    ticker_text = st.sidebar.text_area(
        "Enter tickers separated by commas",
        value="AAPL, MSFT, JPM, NVDA, TSLA, GOOGL, AMZN",
    )

    st.sidebar.subheader("Input Guide")
    st.sidebar.write(
        """
        **Tickers:** Enter multiple public stock symbols separated by commas.

        **Start and end dates:** The historical window used for each stock.

        **Risk-free rate:** Annual rate used in each Sharpe ratio calculation.
        """
    )

    render_screener_page(ticker_text, start_date, end_date, risk_free_rate)
