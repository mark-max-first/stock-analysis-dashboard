"""Pure finance calculations used by the Streamlit dashboard."""

import numpy as np
import pandas as pd


TRADING_DAYS = 252
MIN_OBSERVATIONS = 60
MIN_CAGR_DAYS = 365


def get_close_prices(price_data):
    """Return adjusted closing prices as a clean Series from yfinance output."""
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

    close_prices = close_prices.squeeze()
    if not isinstance(close_prices, pd.Series):
        return pd.Series(dtype=float)
    return close_prices.astype(float).dropna().sort_index()


def calculate_finance_metrics(stock_prices, benchmark_prices, risk_free_rate):
    """Calculate historical return and risk metrics from adjusted daily prices."""
    stock_prices = stock_prices.dropna().astype(float).sort_index()
    benchmark_prices = benchmark_prices.dropna().astype(float).sort_index()
    if len(stock_prices) < MIN_OBSERVATIONS or len(benchmark_prices) < MIN_OBSERVATIONS:
        return None
    if stock_prices.iloc[0] <= 0:
        return None

    stock_returns = stock_prices.pct_change().dropna()
    benchmark_returns = benchmark_prices.pct_change().dropna()
    if stock_returns.empty or benchmark_returns.empty:
        return None

    total_return = float(stock_prices.iloc[-1] / stock_prices.iloc[0] - 1)
    elapsed_days = (stock_prices.index[-1] - stock_prices.index[0]).days
    if elapsed_days <= 0:
        return None
    observed_years = elapsed_days / 365.25
    growth_factor = stock_prices.iloc[-1] / stock_prices.iloc[0]
    cagr = (
        float(growth_factor ** (1 / observed_years) - 1)
        if growth_factor > 0 and elapsed_days >= MIN_CAGR_DAYS
        else np.nan
    )

    daily_volatility = stock_returns.std(ddof=1)
    annualized_volatility = float(daily_volatility * np.sqrt(TRADING_DAYS))
    daily_risk_free_rate = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    if daily_volatility == 0 or np.isnan(daily_volatility):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = float(
            (stock_returns.mean() - daily_risk_free_rate)
            / daily_volatility
            * np.sqrt(TRADING_DAYS)
        )

    aligned_prices = pd.concat(
        [stock_prices.rename("Stock"), benchmark_prices.rename("Benchmark")], axis=1
    ).dropna()
    aligned_returns = aligned_prices.pct_change().dropna()
    benchmark_variance = aligned_returns["Benchmark"].var(ddof=1)
    if (
        len(aligned_prices) < MIN_OBSERVATIONS
        or aligned_returns.empty
        or benchmark_variance == 0
        or np.isnan(benchmark_variance)
    ):
        beta = np.nan
    else:
        beta = float(
            aligned_returns["Stock"].cov(aligned_returns["Benchmark"])
            / benchmark_variance
        )

    running_peak = stock_prices.cummax()
    drawdown = stock_prices / running_peak - 1
    max_drawdown = float(drawdown.min())

    return {
        "stock_returns": stock_returns,
        "benchmark_returns": benchmark_returns,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "beta": beta,
        "drawdown": drawdown,
        "max_drawdown": max_drawdown,
    }


def assign_category(row):
    """Label a stock's historical risk-and-return profile."""
    if row["Max Drawdown"] <= -0.35:
        return "Larger drawdown risk"
    if row["Sharpe Ratio"] >= 1 and row["CAGR"] > 0:
        return "Strong risk-adjusted history"
    if row["CAGR"] >= 0.15 and row["Annualized Volatility"] >= 0.25:
        return "Higher risk / higher return"
    if row["Annualized Volatility"] <= 0.20:
        return "Lower volatility history"
    return "Mixed historical profile"


def add_ranking_score(results):
    """Rank stocks with a transparent weighted percentile score."""
    scored = results.copy()
    sharpe_score = scored["Sharpe Ratio"].rank(pct=True).fillna(0)
    return_score = scored["CAGR"].rank(pct=True).fillna(0)
    drawdown_score = scored["Max Drawdown"].rank(pct=True).fillna(0)
    volatility_score = scored["Annualized Volatility"].rank(
        pct=True, ascending=False
    ).fillna(0)
    scored["Ranking Score"] = (
        sharpe_score * 35
        + return_score * 30
        + drawdown_score * 20
        + volatility_score * 15
    )
    return scored.sort_values(["Ranking Score", "Ticker"], ascending=[False, True])
