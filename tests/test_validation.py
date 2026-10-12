import unittest

from validation import clean_ticker, parse_tickers


class TickerValidationTests(unittest.TestCase):
    def test_clean_ticker_normalizes_supported_symbol(self):
        self.assertEqual(clean_ticker(" brk-b "), "BRK-B")

    def test_clean_ticker_rejects_non_equity_suffix(self):
        self.assertEqual(clean_ticker("EURUSD=X"), "")

    def test_parse_tickers_deduplicates_and_reports_invalid_entries(self):
        valid, invalid = parse_tickers("aapl, MSFT, aapl, BAD!, BRK-B")
        self.assertEqual(valid, ["AAPL", "MSFT", "BRK-B"])
        self.assertEqual(invalid, ["BAD!"])


if __name__ == "__main__":
    unittest.main()
