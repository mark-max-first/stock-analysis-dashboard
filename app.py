from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import PercentFormatter

from finance import (
    MIN_OBSERVATIONS,
    add_ranking_score,
    assign_category,
    calculate_finance_metrics,
    get_close_prices,
)
from validation import clean_ticker, parse_tickers


APP_TITLE = "EquityLens"
BENCHMARK = "SPY"
GITHUB_PROFILE = "https://github.com/mark-max-first"
GITHUB_REPOSITORY = f"{GITHUB_PROFILE}/stock-analysis-dashboard"
MAX_TICKERS = 12

st.set_page_config(
    page_title=f"{APP_TITLE} | Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: #f6f8fb; }
      [data-testid="stSidebar"] { background: #0b1f33; }
      [data-testid="stSidebar"] * { color: #f7fafc; }
      [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce4ec;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 3px 12px rgba(11, 31, 51, 0.05);
      }
      [data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: #52677a !important;
      }
      [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0b1f33 !important;
      }
      [data-testid="stMainBlockContainer"] h1,
      [data-testid="stMainBlockContainer"] h2,
      [data-testid="stMainBlockContainer"] h3 {
        color: #0b1f33;
      }
      .hero {
        background: linear-gradient(135deg, #0b1f33 0%, #123b5d 100%);
        border-radius: 18px;
        color: #ffffff;
        padding: 28px 30px;
        margin-bottom: 20px;
      }
      .hero-kicker {
        color: #63d3c6;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }
      .hero h1 { color: #ffffff; margin: 5px 0 7px; }
      .hero p { color: #d8e4ee; margin: 0; max-width: 780px; }
      .section-note {
        border-left: 4px solid #1d8f83;
        padding: 3px 0 3px 12px;
        color: #425466;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_stock_data(ticker, start_date, end_date):
    """Download adjusted daily price data, including the selected end date."""
    inclusive_end = end_date + timedelta(days=1)
    data = yf.download(
        ticker,
        start=start_date,
        end=inclusive_end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        raise ValueError(f"No market data returned for {ticker}")
    return data


def render_hero(title, description):
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">Portfolio analytics project</div>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def styled_figure(title, ylabel):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color="#0b1f33")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def show_chart(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def format_screener_table(results):
    display_results = results.copy()
    percent = lambda value: "N/A" if pd.isna(value) else f"{value:.2%}"
    number = lambda value: "N/A" if pd.isna(value) else f"{value:.2f}"
    for column in ["Total Return", "CAGR", "Annualized Volatility", "Max Drawdown"]:
        display_results[column] = display_results[column].map(percent)
    for column in ["Sharpe Ratio", "Beta vs SPY"]:
        display_results[column] = display_results[column].map(number)
    display_results["Ranking Score"] = display_results["Ranking Score"].map(
        lambda value: "N/A" if pd.isna(value) else f"{value:.1f}"
    )
    return display_results


def load_prices(ticker, start_date, end_date):
    data = download_stock_data(ticker, start_date, end_date)
    return data, get_close_prices(data)


def render_methodology(risk_free_rate):
    with st.expander("Methodology and limitations"):
        st.markdown(
            f"""
            - **Total return** is the change from the first adjusted close to the last.
            - **CAGR** compounds the beginning and ending adjusted prices over elapsed calendar time
              and is shown only when at least one year of history is available.
            - **Annualized volatility** is the standard deviation of daily returns multiplied by √252.
            - **Sharpe ratio** annualizes average daily excess return using a user-selected
              {risk_free_rate:.2%} annual risk-free rate.
            - **Beta** is daily-return covariance with {BENCHMARK} divided by {BENCHMARK} variance.
            - **Maximum drawdown** is the deepest decline from a previous adjusted-price peak.
            - **Screener scores** are percentile rankings relative only to the submitted list.
              A missing Sharpe ratio receives no Sharpe contribution to the score.

            Prices are adjusted by the data provider for corporate actions. Results are historical,
            depend on the selected window, and are not forecasts or investment advice.
            """
        )


def render_single_stock_page(ticker, start_date, end_date, risk_free_rate):
    render_hero(
        f"{ticker} Stock Analysis",
        f"Evaluate historical return, downside risk, and market sensitivity against {BENCHMARK}.",
    )

    with st.spinner(f"Loading {ticker} and {BENCHMARK} market data..."):
        try:
            _, stock_prices = load_prices(ticker, start_date, end_date)
            _, benchmark_prices = load_prices(BENCHMARK, start_date, end_date)
        except Exception:
            st.error("Market data could not be downloaded. Please try again in a moment.")
            st.stop()

    if stock_prices.empty:
        st.error(f"No adjusted closing prices were found for {ticker}.")
        st.stop()
    if benchmark_prices.empty:
        st.error(f"No benchmark prices were found for {BENCHMARK}.")
        st.stop()

    metrics = calculate_finance_metrics(stock_prices, benchmark_prices, risk_free_rate)
    if metrics is None:
        st.error(
            f"At least {MIN_OBSERVATIONS} adjusted daily prices are required. "
            "Choose a wider date range."
        )
        st.stop()

    actual_start = stock_prices.index.min().date()
    actual_end = stock_prices.index.max().date()
    st.markdown(
        f'<p class="section-note">Adjusted daily prices from <strong>{actual_start}</strong> through '
        f'<strong>{actual_end}</strong>. Benchmark: <strong>{BENCHMARK}</strong>.</p>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cols[0].metric("Latest adjusted close", f"${stock_prices.iloc[-1]:,.2f}")
    cols[1].metric("Total return", f"{metrics['total_return']:.2%}")
    cols[2].metric("CAGR", "N/A" if np.isnan(metrics["cagr"]) else f"{metrics['cagr']:.2%}")
    cols[3].metric("Annualized volatility", f"{metrics['annualized_volatility']:.2%}")

    cols = st.columns(3)
    cols[0].metric(
        "Sharpe ratio",
        "N/A" if np.isnan(metrics["sharpe_ratio"]) else f"{metrics['sharpe_ratio']:.2f}",
    )
    cols[1].metric("Beta vs SPY", "N/A" if np.isnan(metrics["beta"]) else f"{metrics['beta']:.2f}")
    cols[2].metric("Maximum drawdown", f"{metrics['max_drawdown']:.2%}")

    performance_tab, risk_tab, data_tab = st.tabs(["Performance", "Risk", "Data & methodology"])

    with performance_tab:
        aligned_prices = pd.concat(
            [stock_prices.rename(ticker), benchmark_prices.rename(BENCHMARK)], axis=1
        ).dropna()
        growth = aligned_prices / aligned_prices.iloc[0] * 100

        st.subheader("Growth of $100")
        st.caption("Both series begin at $100 on their first shared trading day.")
        fig, ax = styled_figure(f"{ticker} vs {BENCHMARK}", "Hypothetical value ($)")
        ax.plot(growth.index, growth[ticker], label=ticker, color="#0b6e75", linewidth=2.2)
        ax.plot(growth.index, growth[BENCHMARK], label=BENCHMARK, color="#f4a261", linewidth=2.0)
        ax.legend(frameon=False)
        show_chart(fig)

        st.subheader("Trend view")
        st.caption("The 50-day average responds faster; the 200-day average highlights the longer trend.")
        fig, ax = styled_figure(f"{ticker} adjusted close and moving averages", "Price ($)")
        ax.plot(stock_prices.index, stock_prices, label="Adjusted close", color="#0b1f33", linewidth=1.6)
        if len(stock_prices) >= 50:
            ax.plot(
                stock_prices.index,
                stock_prices.rolling(50).mean(),
                label="50-day MA",
                color="#1d8f83",
            )
        if len(stock_prices) >= 200:
            ax.plot(
                stock_prices.index,
                stock_prices.rolling(200).mean(),
                label="200-day MA",
                color="#e76f51",
            )
        ax.legend(frameon=False)
        show_chart(fig)

    with risk_tab:
        st.subheader("Daily return history")
        fig, ax = styled_figure(f"{ticker} daily returns", "Daily return")
        ax.plot(metrics["stock_returns"].index, metrics["stock_returns"], color="#457b9d", linewidth=0.9)
        ax.axhline(0, color="#0b1f33", linewidth=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        show_chart(fig)

        left, right = st.columns(2)
        with left:
            st.subheader("Return distribution")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.hist(metrics["stock_returns"], bins=40, color="#1d8f83", alpha=0.9)
            ax.set_xlabel("Daily return")
            ax.set_ylabel("Trading days")
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))
            ax.spines[["top", "right"]].set_visible(False)
            show_chart(fig)
        with right:
            st.subheader("Drawdown from prior peak")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.fill_between(
                metrics["drawdown"].index,
                metrics["drawdown"].values,
                0,
                color="#e76f51",
                alpha=0.75,
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Drawdown")
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.spines[["top", "right"]].set_visible(False)
            show_chart(fig)

    with data_tab:
        export_data = pd.DataFrame(
            {
                "Adjusted Close": stock_prices,
                "Daily Return": metrics["stock_returns"],
                "Drawdown": metrics["drawdown"],
            }
        )
        export_data.index.name = "Date"
        st.dataframe(export_data.tail(20), use_container_width=True)
        st.download_button(
            "Download analysis data as CSV",
            export_data.to_csv().encode("utf-8"),
            file_name=f"{ticker.lower()}_historical_analysis.csv",
            mime="text/csv",
        )
        render_methodology(risk_free_rate)


def render_screener_page(ticker_text, start_date, end_date, risk_free_rate):
    render_hero(
        "Multi-Stock Screener",
        "Compare a focused list of equities using a transparent historical risk-and-return score.",
    )

    tickers, invalid_tickers = parse_tickers(ticker_text)
    if invalid_tickers:
        st.warning("These ticker entries use unsupported characters: " + ", ".join(invalid_tickers))
    if not tickers:
        st.error("Enter at least one valid ticker.")
        st.stop()
    if len(tickers) > MAX_TICKERS:
        st.error(f"Enter no more than {MAX_TICKERS} tickers at a time.")
        st.stop()

    try:
        _, benchmark_prices = load_prices(BENCHMARK, start_date, end_date)
    except Exception:
        st.error("Benchmark data could not be downloaded. Please try again in a moment.")
        st.stop()
    if benchmark_prices.empty:
        st.error(f"No benchmark prices were found for {BENCHMARK}.")
        st.stop()

    rows, skipped, shorter_histories = [], [], []
    progress = st.progress(0, text="Analyzing tickers...")
    for index, ticker in enumerate(tickers, start=1):
        try:
            _, prices = load_prices(ticker, start_date, end_date)
            metrics = calculate_finance_metrics(prices, benchmark_prices, risk_free_rate)
        except Exception:
            metrics = None

        if metrics is None:
            skipped.append(ticker)
        else:
            first_date = prices.index.min().date()
            if first_date > start_date + timedelta(days=5):
                shorter_histories.append(f"{ticker} ({first_date})")
            rows.append(
                {
                    "Ticker": ticker,
                    "First Date": first_date.isoformat(),
                    "Observations": len(prices),
                    "Total Return": metrics["total_return"],
                    "CAGR": metrics["cagr"],
                    "Annualized Volatility": metrics["annualized_volatility"],
                    "Sharpe Ratio": metrics["sharpe_ratio"],
                    "Beta vs SPY": metrics["beta"],
                    "Max Drawdown": metrics["max_drawdown"],
                }
            )
        progress.progress(index / len(tickers), text=f"Analyzed {index} of {len(tickers)} tickers")
    progress.empty()

    if not rows:
        st.error("No usable price history was found for the submitted tickers.")
        st.stop()

    results = add_ranking_score(pd.DataFrame(rows))
    results["Profile"] = results.apply(assign_category, axis=1)
    columns = [
        "Ticker",
        "First Date",
        "Observations",
        "Total Return",
        "CAGR",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Beta vs SPY",
        "Max Drawdown",
        "Ranking Score",
        "Profile",
    ]

    st.subheader("Ranked results")
    st.caption(
        f"Requested period: {start_date} through {end_date}. The score weights Sharpe ratio 35%, CAGR 30%, "
        "maximum drawdown 20%, and volatility 15%."
    )
    st.caption("Scores are relative only to the submitted list and cannot be compared across separate runs.")
    if len(results) < 3:
        st.warning("Use at least three valid tickers for a more informative relative ranking.")
    st.dataframe(format_screener_table(results[columns]), use_container_width=True, hide_index=True)
    st.download_button(
        "Download screener results as CSV",
        results[columns].to_csv(index=False).encode("utf-8"),
        file_name="stock_screener_results.csv",
        mime="text/csv",
    )
    if skipped:
        st.warning("Skipped because data was unavailable or incomplete: " + ", ".join(skipped))
    if shorter_histories:
        st.warning(
            "These tickers have less history than the requested window: "
            + ", ".join(shorter_histories)
        )
    st.info("The ranking is descriptive, not predictive. It is not a recommendation to buy or sell.")
    render_methodology(risk_free_rate)


st.sidebar.markdown(f"## 📈 {APP_TITLE}")
st.sidebar.caption("Historical equity risk and return analysis")
page = st.sidebar.radio("Analysis", ["Single stock", "Stock screener"])

st.sidebar.markdown("### Analysis period")
default_end_date = date.today()
default_start_date = default_end_date - timedelta(days=365 * 5)
start_date = st.sidebar.date_input("Start date", value=default_start_date)
end_date = st.sidebar.date_input("End date", value=default_end_date)
risk_free_rate = st.sidebar.number_input(
    "Annual risk-free rate (%)",
    min_value=0.0,
    max_value=25.0,
    value=4.0,
    step=0.25,
    help="User-selected assumption used only in the Sharpe ratio.",
) / 100

if start_date >= end_date:
    st.error("The start date must be earlier than the end date.")
    st.stop()

if page == "Single stock":
    raw_ticker = st.sidebar.text_input("Ticker", value="AAPL", help="Examples: AAPL, JPM, BRK-B")
    ticker = clean_ticker(raw_ticker)
    if not ticker:
        st.error("Enter a valid US-listed equity or ETF ticker using letters, numbers, periods, or hyphens.")
        st.stop()
    render_single_stock_page(ticker, start_date, end_date, risk_free_rate)
else:
    ticker_text = st.sidebar.text_area(
        "Tickers separated by commas",
        value="AAPL, MSFT, JPM, NVDA, GOOGL",
        help=f"Up to {MAX_TICKERS} unique tickers.",
    )
    render_screener_page(ticker_text, start_date, end_date, risk_free_rate)

st.sidebar.divider()
st.sidebar.link_button("View source on GitHub", GITHUB_REPOSITORY, use_container_width=True)
st.sidebar.link_button("GitHub profile", GITHUB_PROFILE, use_container_width=True)
st.sidebar.caption("Built by Mark Spinelli · Educational use only")
