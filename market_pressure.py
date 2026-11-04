"""15-minute Binance spot / USDT-M taker-flow pressure features.

A positive value means more taker BUY notional than taker SELL notional.
These are executed-trade imbalances, NOT net money inflow, investor position
imbalance, directional forecasts, or the existing LS/OI heat score.

The first integration deliberately supports only matching Binance USDT spot
and USDT-M perpetual pairs. COIN-M and other quote assets need explicit
contract-size and currency normalization before joining this calculation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_trade_pressure_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-pair 15m pressures without imputing unavailable market data.

    Expected optional inputs for each market 'spot' / 'futures':
        <market>_quote_volume_15m: total USDT traded in the 15m kline
        <market>_taker_buy_quote_volume_15m: taker-buy USDT in that kline

    A partial combined value contains ONLY available, valid markets. Inspect
    trade_pressure_coverage before treating it as a two-market reading.
    """
    out = frame.copy()
    valid_volumes: dict[str, pd.Series] = {}
    valid_nets: dict[str, pd.Series] = {}
    pressures: dict[str, pd.Series] = {}

    for market in ("spot", "futures"):
        volume_col = f"{market}_quote_volume_15m"
        buy_col = f"{market}_taker_buy_quote_volume_15m"
        volume = pd.to_numeric(
            out[volume_col] if volume_col in out else pd.Series(np.nan, index=out.index),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        buy = pd.to_numeric(
            out[buy_col] if buy_col in out else pd.Series(np.nan, index=out.index),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        # Zero-turnover bars carry no directional observation. Reject impossible
        # buy-notional values instead of silently clipping invalid API data.
        valid = volume.gt(0) & buy.ge(0) & buy.le(volume)
        accepted_volume = volume.where(valid)
        sell = (volume - buy).where(valid)
        net = (buy - sell).where(valid)
        pressure = (100.0 * net / accepted_volume).where(valid)

        out[f"{market}_taker_sell_quote_volume_15m"] = sell
        out[f"{market}_net_taker_quote_volume_15m"] = net
        out[f"{market}_trade_pressure_pct"] = pressure
        valid_volumes[market] = accepted_volume
        valid_nets[market] = net
        pressures[market] = pressure

    spot_valid = valid_volumes["spot"].notna()
    futures_valid = valid_volumes["futures"].notna()
    gross = valid_volumes["spot"].fillna(0) + valid_volumes["futures"].fillna(0)
    net = valid_nets["spot"].fillna(0) + valid_nets["futures"].fillna(0)
    has_market = gross.gt(0)

    out["trade_pressure_quote_asset"] = "USDT"
    out["trade_pressure_scope"] = "BINANCE_SPOT_USDTM_PERPETUAL"
    out["trade_pressure_coverage"] = np.select(
        [spot_valid & futures_valid, spot_valid, futures_valid],
        ["SPOT_AND_USDTM", "SPOT_ONLY", "USDTM_ONLY"],
        default="NO_VALID_MARKET",
    )
    out["trade_pressure_market_count"] = spot_valid.astype(int) + futures_valid.astype(int)
    out["trade_pressure_total_quote_volume_15m"] = gross.where(has_market)
    out["trade_pressure_net_taker_quote_volume_15m"] = net.where(has_market)
    out["trade_pressure_turnover_weighted_pct"] = (100.0 * net / gross.where(has_market))
    out["spot_trade_volume_share_pct"] = (
        100.0 * valid_volumes["spot"].fillna(0) / gross.where(has_market)
    )
    out["futures_trade_volume_share_pct"] = (
        100.0 * valid_volumes["futures"].fillna(0) / gross.where(has_market)
    )
    out["trade_pressure_balanced_pct"] = (
        (pressures["spot"] + pressures["futures"]) / 2.0
    ).where(spot_valid & futures_valid)
    return out
