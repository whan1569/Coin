"""Public Binance spot/USDT-M closed-candle pressure snapshot collector.

No user credentials, order endpoints, or account data. Binance public market
data is IP rate limited and may not be reachable in some jurisdictions.
Commercial redistribution rights must be confirmed separately.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd

from market_pressure import add_trade_pressure_features

SPOT = "https://api.binance.com"
USDT_M = "https://fapi.binance.com"
PERIOD_MS = 15 * 60 * 1000
HISTORY_COLUMNS = [
    "symbol", "timestamp_ms", "futures_close",
    "spot_quote_volume_15m", "spot_taker_buy_quote_volume_15m",
    "futures_quote_volume_15m", "futures_taker_buy_quote_volume_15m",
]
DATA_FILE = (
    Path(os.getenv("LOCALAPPDATA", str(Path.home() / ".local" / "share")))
    / "GMT" / "Coin" / "public_pressure_history.csv"
)


def most_recent_closed_open_ms(server_ms: int) -> int:
    """Open time for the most recent fully closed 15-minute UTC candle."""
    if server_ms < PERIOD_MS:
        raise ValueError("Invalid server time")
    return (server_ms // PERIOD_MS - 1) * PERIOD_MS


def parse_closed_kline(value: Any, target_open_ms: int) -> dict[str, float] | None:
    """Accept one exact closed candle; do not accidentally mix unequal bars."""
    if not isinstance(value, list) or not value or not isinstance(value[0], (list, tuple)):
        return None
    bar = value[0]
    if len(bar) < 11 or int(bar[0]) != target_open_ms:
        return None
    if int(bar[6]) < target_open_ms + PERIOD_MS - 1:
        return None
    total = float(bar[7])
    taker_buy = float(bar[10])
    close = float(bar[4])
    if not all(np.isfinite(v) for v in (total, taker_buy, close)):
        return None
    if not (total > 0 and 0 <= taker_buy <= total and close > 0):
        return None
    return {"quote_volume": total, "taker_buy_quote_volume": taker_buy, "close": close}


async def _get_json(
    session: aiohttp.ClientSession, url: str,
    params: dict | None = None, retries: int = 3,
) -> Any:
    for attempt in range(retries):
        try:
            async with session.get(url, params=params) as response:
                if response.status in (418, 429):
                    if attempt == retries - 1:
                        raise RuntimeError("Binance request limit exceeded (418/429). Try again later.")
                    delay = min(float(response.headers.get("Retry-After", 2 ** (attempt + 1))), 30.0)
                    await asyncio.sleep(max(0.5, delay))
                    continue
                if response.status in (403, 451):
                    raise RuntimeError(
                        f"Binance HTTP {response.status}: API access restricted for this network or region."
                    )
                if response.status >= 500:
                    if attempt == retries - 1:
                        raise RuntimeError(f"Binance HTTP {response.status}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError("Binance request retries exhausted")


def select_usdt_perpetual_symbols(
    futures_info: dict, spot_info: dict, tickers: list[dict],
    limit: int = 40,
) -> list[str]:
    """Sort active, spot-matched USDT perpetual markets by 24h futures turnover."""
    futures = {
        row["symbol"] for row in futures_info.get("symbols", [])
        if row.get("contractType") == "PERPETUAL"
        and row.get("quoteAsset") == "USDT"
        and row.get("status") == "TRADING"
    }
    spot = {
        row["symbol"] for row in spot_info.get("symbols", [])
        if row.get("quoteAsset") == "USDT"
        and row.get("status") == "TRADING"
        and row.get("isSpotTradingAllowed", True)
    }
    available = futures & spot
    volumes = []
    for ticker in tickers:
        symbol = ticker.get("symbol")
        if symbol not in available:
            continue
        try:
            volume = float(ticker.get("quoteVolume", 0))
        except (TypeError, ValueError):
            continue
        if np.isfinite(volume) and volume > 0:
            volumes.append((volume, symbol))
    volumes.sort(key=lambda pair: (-pair[0], pair[1]))
    return [symbol for _, symbol in volumes[:max(1, min(int(limit), 100))]]


async def collect_public_pressure(
    limit: int = 40,
    concurrency: int = 8,
) -> tuple[pd.DataFrame, dict]:
    """Fetch a single completed 15m bar per market; partial markets are labeled."""
    timeout = aiohttp.ClientTimeout(total=18, connect=7, sock_read=12)
    connector = aiohttp.TCPConnector(limit=max(1, min(concurrency, 12)))
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector,
        headers={"User-Agent": "GMT-Coin/0.1 market-observation"},
    ) as session:
        futures_info, spot_info, tickers, clock = await asyncio.gather(
            _get_json(session, f"{USDT_M}/fapi/v1/exchangeInfo"),
            _get_json(session, f"{SPOT}/api/v3/exchangeInfo"),
            _get_json(session, f"{USDT_M}/fapi/v1/ticker/24hr"),
            _get_json(session, f"{USDT_M}/fapi/v1/time"),
        )
        symbols = select_usdt_perpetual_symbols(
            futures_info, spot_info, tickers, limit=limit
        )
        if not symbols:
            raise RuntimeError("No Binance spot-matched USDT perpetual pairs were discovered.")
        target = most_recent_closed_open_ms(int(clock["serverTime"]))
        semaphore = asyncio.Semaphore(max(1, min(concurrency, 12)))
        failed = []

        async def read_one(symbol: str) -> dict:
            query = {
                "symbol": symbol, "interval": "15m",
                "startTime": target,
                "endTime": target + PERIOD_MS - 1,
                "limit": 1,
            }

            async def one_market(market: str, url: str) -> dict | None:
                async with semaphore:
                    try:
                        data = await _get_json(session, url, query)
                        parsed = parse_closed_kline(data, target)
                        if parsed is None:
                            failed.append(f"{symbol}/{market}: unavailable or wrong candle")
                        return parsed
                    except Exception as exc:
                        failed.append(f"{symbol}/{market}: {type(exc).__name__}: {str(exc)[:100]}")
                        return None

            spot_bar, futures_bar = await asyncio.gather(
                one_market("spot", f"{SPOT}/api/v3/klines"),
                one_market("usdtm", f"{USDT_M}/fapi/v1/klines"),
            )
            record = {"symbol": symbol, "timestamp_ms": target}
            if futures_bar:
                record["futures_quote_volume_15m"] = futures_bar["quote_volume"]
                record["futures_taker_buy_quote_volume_15m"] = futures_bar["taker_buy_quote_volume"]
                record["futures_close"] = futures_bar["close"]
            if spot_bar:
                record["spot_quote_volume_15m"] = spot_bar["quote_volume"]
                record["spot_taker_buy_quote_volume_15m"] = spot_bar["taker_buy_quote_volume"]
            return record

        rows = await asyncio.gather(*(read_one(symbol) for symbol in symbols))
    raw = pd.DataFrame(rows)
    for column in HISTORY_COLUMNS:
        if column not in raw.columns:
            raw[column] = np.nan
    out = add_trade_pressure_features(raw[HISTORY_COLUMNS])
    usable = out["trade_pressure_market_count"].gt(0)
    out = out[usable].reset_index(drop=True)
    if out.empty:
        raise RuntimeError(
            "Binance returned no usable completed candles. "
            + ("; ".join(failed[:3]) if failed else "")
        )
    meta = {
        "requested": len(symbols),
        "returned": len(out),
        "failures": len(failed),
        "samples": failed[:6],
        "candle_open_ms": target,
        "scope": "BINANCE_SPOT_USDTM_PERPETUAL",
    }
    return out, meta


def load_local_history(path: Path = DATA_FILE) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    frame = pd.read_csv(path)
    if not {"symbol", "timestamp_ms"}.issubset(frame.columns):
        raise ValueError("Stored history lacks symbol or timestamp_ms; please back it up and inspect.")
    return frame


def save_local_history(
    latest: pd.DataFrame, path: Path = DATA_FILE, retain_days: int = 120,
) -> pd.DataFrame:
    """Save raw trades atomically; never erase data after a failed request."""
    if latest.empty:
        return load_local_history(path)
    existing = load_local_history(path)
    raw = pd.concat(
        [existing, latest.reindex(columns=HISTORY_COLUMNS)],
        ignore_index=True,
    )
    raw["timestamp_ms"] = pd.to_numeric(raw["timestamp_ms"], errors="coerce")
    raw = raw.dropna(subset=["symbol", "timestamp_ms"])
    raw = raw.sort_values(["timestamp_ms", "symbol"]).drop_duplicates(
        ["symbol", "timestamp_ms"], keep="last"
    )
    cutoff = int(time.time() * 1000) - retain_days * 86_400_000
    raw = raw[raw["timestamp_ms"] >= cutoff]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        raw.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return raw
