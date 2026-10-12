import unittest

import numpy as np
import pandas as pd

from finance import (
    MIN_OBSERVATIONS,
    add_ranking_score,
    calculate_finance_metrics,
    get_close_prices,
)


class FinanceMetricsTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2025-01-02", periods=MIN_OBSERVATIONS)

    def test_total_return_and_max_drawdown(self):
        prices = [100, 110, 99] + list(np.linspace(100, 120, MIN_OBSERVATIONS - 4)) + [114]
        stock = pd.Series(prices, index=self.dates)
        benchmark = pd.Series(np.linspace(100, 110, MIN_OBSERVATIONS), index=self.dates)
        metrics = calculate_finance_metrics(stock, benchmark, 0.04)
        self.assertAlmostEqual(metrics["total_return"], 0.14)
        self.assertAlmostEqual(metrics["max_drawdown"], -0.10)

    def test_beta_uses_returns_from_shared_price_dates(self):
        dates = pd.bdate_range("2025-01-02", periods=80)
        benchmark_returns = np.resize(
            np.array([0.01, -0.005, 0.02, -0.01, 0.015]), len(dates) - 1
        )
        stock_returns = benchmark_returns * 1.5
        benchmark = pd.Series(
            100 * np.cumprod(np.r_[1.0, 1 + benchmark_returns]), index=dates
        )
        stock = pd.Series(
            100 * np.cumprod(np.r_[1.0, 1 + stock_returns]), index=dates
        )
        metrics = calculate_finance_metrics(stock, benchmark, 0.0)
        self.assertAlmostEqual(metrics["beta"], 1.5, places=10)

    def test_cagr_uses_elapsed_calendar_time(self):
        dates = pd.date_range("2024-01-01", "2026-01-01", periods=MIN_OBSERVATIONS)
        stock = pd.Series(np.geomspace(100, 121, MIN_OBSERVATIONS), index=dates)
        benchmark = pd.Series(np.geomspace(100, 115, MIN_OBSERVATIONS), index=dates)
        expected = (121 / 100) ** (365.25 / 731) - 1
        metrics = calculate_finance_metrics(stock, benchmark, 0.0)
        self.assertAlmostEqual(metrics["cagr"], expected, places=12)

    def test_volatility_and_sharpe_match_daily_formula(self):
        dates = pd.bdate_range("2025-01-02", periods=80)
        returns = np.resize(np.array([0.01, -0.004, 0.006, -0.002]), len(dates) - 1)
        stock = pd.Series(100 * np.cumprod(np.r_[1.0, 1 + returns]), index=dates)
        benchmark = pd.Series(np.linspace(100, 110, len(dates)), index=dates)
        risk_free_rate = 0.04
        expected_volatility = returns.std(ddof=1) * np.sqrt(252)
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        expected_sharpe = (returns.mean() - daily_rf) / returns.std(ddof=1) * np.sqrt(252)
        metrics = calculate_finance_metrics(stock, benchmark, risk_free_rate)
        self.assertAlmostEqual(metrics["annualized_volatility"], expected_volatility)
        self.assertAlmostEqual(metrics["sharpe_ratio"], expected_sharpe)

    def test_flat_prices_produce_zero_volatility_and_no_sharpe(self):
        flat = pd.Series([100.0] * MIN_OBSERVATIONS, index=self.dates)
        benchmark = pd.Series(np.linspace(100, 110, MIN_OBSERVATIONS), index=self.dates)
        metrics = calculate_finance_metrics(flat, benchmark, 0.04)
        self.assertEqual(metrics["annualized_volatility"], 0.0)
        self.assertTrue(np.isnan(metrics["sharpe_ratio"]))
        self.assertEqual(metrics["max_drawdown"], 0.0)

    def test_short_series_is_rejected(self):
        dates = pd.bdate_range("2025-01-02", periods=MIN_OBSERVATIONS - 1)
        prices = pd.Series(np.linspace(100, 110, len(dates)), index=dates)
        self.assertIsNone(calculate_finance_metrics(prices, prices, 0.04))

    def test_cagr_is_not_shown_for_less_than_one_year(self):
        dates = pd.date_range("2025-01-01", "2025-06-30", periods=MIN_OBSERVATIONS)
        stock = pd.Series(np.linspace(100, 120, MIN_OBSERVATIONS), index=dates)
        benchmark = pd.Series(np.linspace(100, 110, MIN_OBSERVATIONS), index=dates)
        metrics = calculate_finance_metrics(stock, benchmark, 0.04)
        self.assertTrue(np.isnan(metrics["cagr"]))

    def test_get_close_prices_handles_yfinance_multiindex(self):
        columns = pd.MultiIndex.from_tuples([("Close", "AAPL"), ("Volume", "AAPL")])
        data = pd.DataFrame([[100.0, 1000], [101.0, 1100]], columns=columns)
        close = get_close_prices(data)
        pd.testing.assert_series_equal(close, pd.Series([100.0, 101.0]), check_names=False)

    def test_ranking_score_rewards_stronger_combination(self):
        results = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "CAGR": 0.18,
                    "Annualized Volatility": 0.18,
                    "Sharpe Ratio": 1.2,
                    "Max Drawdown": -0.15,
                },
                {
                    "Ticker": "BBB",
                    "CAGR": 0.08,
                    "Annualized Volatility": 0.32,
                    "Sharpe Ratio": 0.3,
                    "Max Drawdown": -0.40,
                },
            ]
        )
        ranked = add_ranking_score(results)
        self.assertEqual(ranked.iloc[0]["Ticker"], "AAA")
        self.assertGreater(ranked.iloc[0]["Ranking Score"], ranked.iloc[1]["Ranking Score"])


if __name__ == "__main__":
    unittest.main()
