"""Offline synthetic tests; no Binance network or API credentials required."""
import unittest

import numpy as np
import pandas as pd

from market_pressure import add_trade_pressure_features


class TradePressureTests(unittest.TestCase):
    def test_weighted_and_balanced_are_different(self):
        df = pd.DataFrame([{
            "symbol": "BTCUSDT",
            "spot_quote_volume_15m": 100.0,
            "spot_taker_buy_quote_volume_15m": 60.0,  # +20%
            "futures_quote_volume_15m": 900.0,
            "futures_taker_buy_quote_volume_15m": 360.0,  # -20%
        }])
        row = add_trade_pressure_features(df).iloc[0]
        self.assertAlmostEqual(row["spot_trade_pressure_pct"], 20.0)
        self.assertAlmostEqual(row["futures_trade_pressure_pct"], -20.0)
        self.assertAlmostEqual(row["trade_pressure_turnover_weighted_pct"], -16.0)
        self.assertAlmostEqual(row["trade_pressure_balanced_pct"], 0.0)
        self.assertAlmostEqual(row["spot_trade_volume_share_pct"], 10.0)
        self.assertEqual(row["trade_pressure_coverage"], "SPOT_AND_USDTM")
        self.assertEqual(row["trade_pressure_quote_asset"], "USDT")

    def test_missing_spot_is_not_treated_as_neutral_observation(self):
        df = pd.DataFrame([{
            "futures_quote_volume_15m": 50.0,
            "futures_taker_buy_quote_volume_15m": 10.0,
        }])
        row = add_trade_pressure_features(df).iloc[0]
        self.assertEqual(row["trade_pressure_coverage"], "USDTM_ONLY")
        self.assertEqual(row["trade_pressure_market_count"], 1)
        self.assertAlmostEqual(row["trade_pressure_turnover_weighted_pct"], -60.0)
        self.assertTrue(np.isnan(row["trade_pressure_balanced_pct"]))
        self.assertTrue(np.isnan(row["spot_trade_pressure_pct"]))

    def test_missing_or_invalid_values_are_not_fabricated(self):
        df = pd.DataFrame([
            {"spot_quote_volume_15m": 0, "spot_taker_buy_quote_volume_15m": 0},
            {"spot_quote_volume_15m": 100, "spot_taker_buy_quote_volume_15m": 120},
            {"spot_quote_volume_15m": np.inf, "spot_taker_buy_quote_volume_15m": 4},
        ])
        out = add_trade_pressure_features(df)
        self.assertTrue(out["trade_pressure_turnover_weighted_pct"].isna().all())
        self.assertTrue((out["trade_pressure_coverage"] == "NO_VALID_MARKET").all())

    def test_vectorized_independent_rows_and_no_input_mutation(self):
        df = pd.DataFrame([
            {"spot_quote_volume_15m": 20, "spot_taker_buy_quote_volume_15m": 20,
             "futures_quote_volume_15m": 20, "futures_taker_buy_quote_volume_15m": 0},
            {"spot_quote_volume_15m": 20, "spot_taker_buy_quote_volume_15m": 10,
             "futures_quote_volume_15m": 20, "futures_taker_buy_quote_volume_15m": 10},
        ], index=[5, 10])
        out = add_trade_pressure_features(df)
        self.assertEqual(list(out.index), [5, 10])
        self.assertEqual(list(out["trade_pressure_turnover_weighted_pct"]), [0.0, 0.0])
        self.assertNotIn("trade_pressure_scope", df)


if __name__ == "__main__":
    unittest.main()
