# LS_history_collector.py

import argparse
import asyncio
import os
import time
from functools import reduce
from typing import Optional

import aiohttp
import numpy as np
import pandas as pd


# =========================================================
# 설정
# =========================================================

BASE_FAPI = "https://fapi.binance.com"
BASE_DATA = "https://fapi.binance.com/futures/data"
BASE_SPOT = "https://api.binance.com"

PERIOD = "15m"
PERIOD_MS = 15 * 60 * 1000

DEFAULT_DAYS = 20
DEFAULT_CONCURRENCY = 15
DEFAULT_OUTPUT = "data/history_lsoi_15m.csv"

EXCLUDE_SYMBOLS = set()

WEIGHTS = {
    "ls_ratio": 0.35,
    "ls_acco": 0.15,
    "ls_position": 0.50,
}

ENDPOINTS = {
    "ls_ratio": "globalLongShortAccountRatio",
    "ls_acco": "topLongShortAccountRatio",
    "ls_position": "topLongShortPositionRatio",
}

RAW_REQUIRED_COLS = [
    "ls_ratio",
    "ls_acco",
    "ls_position",
    "open_interest",
    "oi_nv",
    "mark_price",
]

FUNDING_FEATURE_COLS = [
    "funding_rate_pct",
    "funding_interval_hours",
    "funding_rate_8h",
    "funding_rate_8h_pct",
    "funding_daily_pct",
    "funding_abs_8h_pct",
]


# =========================================================
# 공통 유틸
# =========================================================

def now_ms() -> int:
    return int(time.time() * 1000)


def floor_to_period_ms(ts_ms: int, period_ms: int = PERIOD_MS) -> int:
    return ts_ms - (ts_ms % period_ms)


def ms_to_utc_str(ts_ms: int) -> str:
    return pd.to_datetime(ts_ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)

    if parent:
        os.makedirs(parent, exist_ok=True)


def safe_float(value) -> float:
    try:
        if value is None:
            return np.nan

        return float(value)

    except Exception:
        return np.nan


def safe_q90(series: pd.Series) -> float:
    value = series.replace([np.inf, -np.inf], np.nan).dropna().quantile(0.90)

    if pd.isna(value) or value <= 0:
        return 1.0

    return float(value)


def normalize_by_timestamp(
    df: pd.DataFrame,
    source_col: str,
    cap: float = 5.0,
) -> pd.Series:
    q90 = df.groupby("timestamp_ms")[source_col].transform(safe_q90)
    out = df[source_col] / q90

    return out.clip(lower=0, upper=cap)


def expected_timestamps(start_ms: int, end_ms: int) -> list[int]:
    if start_ms > end_ms:
        return []

    return list(range(start_ms, end_ms + PERIOD_MS, PERIOD_MS))


def group_consecutive_timestamps(timestamps: list[int]) -> list[tuple[int, int]]:
    if not timestamps:
        return []

    timestamps = sorted(set(int(x) for x in timestamps))

    ranges = []
    start = timestamps[0]
    prev = timestamps[0]

    for ts in timestamps[1:]:
        if ts == prev + PERIOD_MS:
            prev = ts
            continue

        ranges.append((start, prev))
        start = ts
        prev = ts

    ranges.append((start, prev))

    return ranges


def normalize_existing_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()

    if "timestamp_ms" in out.columns:
        out["timestamp_ms"] = pd.to_numeric(out["timestamp_ms"], errors="coerce")
        out = out.dropna(subset=["timestamp_ms"]).copy()
        out["timestamp_ms"] = out["timestamp_ms"].astype("int64")

    numeric_cols = (
        RAW_REQUIRED_COLS
        + [
            "funding_time_ms",
            "funding_rate",
            "funding_mark_price",
            "spot_quote_volume_24h",
            "spot_quote_volume_24h_current",
            "spot_quote_volume_15m",
            "spot_close",
        ]
        + FUNDING_FEATURE_COLS
    )

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def get_missing_ranges_for_symbol(
    existing_df: pd.DataFrame,
    symbol: str,
    start_keep_ms: int,
    end_ms: int,
) -> list[tuple[int, int]]:
    all_ts = expected_timestamps(start_keep_ms, end_ms)

    if not all_ts:
        return []

    if existing_df is None or existing_df.empty:
        return [(start_keep_ms, end_ms)]

    if "symbol" not in existing_df.columns or "timestamp_ms" not in existing_df.columns:
        return [(start_keep_ms, end_ms)]

    sub = existing_df[existing_df["symbol"] == symbol].copy()

    if sub.empty:
        return [(start_keep_ms, end_ms)]

    valid_mask = pd.Series(True, index=sub.index)

    for col in RAW_REQUIRED_COLS:
        if col not in sub.columns:
            valid_mask &= False
        else:
            valid_mask &= sub[col].notna()

    valid_sub = sub[valid_mask].copy()

    existing_ts = set(
        pd.to_numeric(valid_sub["timestamp_ms"], errors="coerce")
        .dropna()
        .astype("int64")
        .tolist()
    )

    missing_ts = [ts for ts in all_ts if ts not in existing_ts]

    return group_consecutive_timestamps(missing_ts)


# =========================================================
# HTTP
# =========================================================

async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[dict] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    retries: int = 4,
):
    last_error = None

    for attempt in range(retries):
        try:
            if semaphore is None:
                async with session.get(url, params=params) as response:
                    if response.status in (418, 429):
                        await asyncio.sleep(2.0 * (attempt + 1))
                        continue

                    response.raise_for_status()
                    return await response.json()

            async with semaphore:
                async with session.get(url, params=params) as response:
                    if response.status in (418, 429):
                        await asyncio.sleep(2.0 * (attempt + 1))
                        continue

                    response.raise_for_status()
                    return await response.json()

        except Exception as exc:
            last_error = exc
            await asyncio.sleep(0.5 * (attempt + 1))

    raise RuntimeError(f"fetch failed: {url}, params={params}, error={last_error}")


async def fetch_paginated(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    base_params: dict,
    start_ms: int,
    end_ms: int,
    ts_key: str,
    limit: int = 500,
) -> list[dict]:
    rows = []

    chunk_bars = limit - 1
    chunk_ms = chunk_bars * PERIOD_MS

    cursor = start_ms

    while cursor <= end_ms:
        chunk_end = min(cursor + chunk_ms, end_ms)

        params = dict(base_params)
        params.update(
            {
                "startTime": cursor,
                "endTime": chunk_end,
                "limit": limit,
            }
        )

        data = await fetch_json(
            session=session,
            url=url,
            params=params,
            semaphore=semaphore,
        )

        if data:
            rows.extend(data)

        cursor = chunk_end + PERIOD_MS

    dedup = {}

    for item in rows:
        try:
            ts = int(item[ts_key])
            dedup[ts] = item
        except Exception:
            continue

    return [dedup[ts] for ts in sorted(dedup.keys())]


# =========================================================
# 심볼 / 현물 정보
# =========================================================

async def get_usdt_perp_symbols(session: aiohttp.ClientSession) -> list[str]:
    url = f"{BASE_FAPI}/fapi/v1/exchangeInfo"
    data = await fetch_json(session, url)

    symbols = []

    for item in data.get("symbols", []):
        if item.get("contractType") != "PERPETUAL":
            continue

        if item.get("quoteAsset") != "USDT":
            continue

        if item.get("status") != "TRADING":
            continue

        symbol = item.get("symbol")

        if not symbol or symbol in EXCLUDE_SYMBOLS:
            continue

        symbols.append(symbol)

    return sorted(symbols)


async def get_spot_quote_volume_map(session: aiohttp.ClientSession) -> dict[str, float]:
    url = f"{BASE_SPOT}/api/v3/ticker/24hr"
    data = await fetch_json(session, url)

    out = {}

    for item in data:
        symbol = str(item.get("symbol", "")).upper().strip()
        quote_volume = safe_float(item.get("quoteVolume"))

        if symbol and quote_volume > 0:
            out[symbol] = quote_volume

    return out


def spot_symbol_candidates(futures_symbol: str) -> list[str]:
    candidates = [futures_symbol]

    if futures_symbol.endswith("USDT"):
        base = futures_symbol[:-4]

        for prefix in ["1000000", "100000", "10000", "1000", "100"]:
            if base.startswith(prefix):
                candidates.append(base[len(prefix):] + "USDT")

    return list(dict.fromkeys(candidates))


def find_spot_info(
    futures_symbol: str,
    spot_quote_volume_map: dict[str, float],
) -> dict:
    for candidate in spot_symbol_candidates(futures_symbol):
        quote_volume = spot_quote_volume_map.get(candidate)

        if quote_volume is not None and quote_volume > 0:
            return {
                "symbol": futures_symbol,
                "spot_symbol": candidate,
                "spot_quote_volume_24h_current": quote_volume,
                "spot_market_category": "SPOT_OK",
            }

    return {
        "symbol": futures_symbol,
        "spot_symbol": None,
        "spot_quote_volume_24h_current": np.nan,
        "spot_market_category": "SPOT_MISSING",
    }


# =========================================================
# Futures 데이터 수집
# =========================================================

async def fetch_ratio_history(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol: str,
    key: str,
    endpoint: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    url = f"{BASE_DATA}/{endpoint}"

    data = await fetch_paginated(
        session=session,
        semaphore=semaphore,
        url=url,
        base_params={"symbol": symbol, "period": PERIOD},
        start_ms=start_ms,
        end_ms=end_ms,
        ts_key="timestamp",
        limit=500,
    )

    rows = []

    for item in data:
        rows.append(
            {
                "symbol": symbol,
                "timestamp_ms": int(item["timestamp"]),
                key: safe_float(item.get("longShortRatio")),
            }
        )

    return pd.DataFrame(rows)


async def fetch_open_interest_history(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    url = f"{BASE_DATA}/openInterestHist"

    data = await fetch_paginated(
        session=session,
        semaphore=semaphore,
        url=url,
        base_params={"symbol": symbol, "period": PERIOD},
        start_ms=start_ms,
        end_ms=end_ms,
        ts_key="timestamp",
        limit=500,
    )

    rows = []

    for item in data:
        open_interest = safe_float(item.get("sumOpenInterest"))
        oi_nv = safe_float(item.get("sumOpenInterestValue"))
        mark_price = oi_nv / open_interest if open_interest and open_interest > 0 else np.nan

        rows.append(
            {
                "symbol": symbol,
                "timestamp_ms": int(item["timestamp"]),
                "open_interest": open_interest,
                "oi_nv": oi_nv,
                "mark_price": mark_price,
            }
        )

    return pd.DataFrame(rows)


async def fetch_funding_history(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    url = f"{BASE_FAPI}/fapi/v1/fundingRate"
    rows = []

    cursor = max(0, start_ms - 8 * 60 * 60 * 1000)

    while cursor <= end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }

        data = await fetch_json(
            session=session,
            url=url,
            params=params,
            semaphore=semaphore,
        )

        if not data:
            break

        for item in data:
            rows.append(
                {
                    "symbol": symbol,
                    "funding_time_ms": int(item["fundingTime"]),
                    "funding_rate": safe_float(item.get("fundingRate")),
                    "funding_mark_price": safe_float(item.get("markPrice")),
                }
            )

        last_ts = int(data[-1]["fundingTime"])
        cursor = last_ts + 1

        if len(data) < 1000:
            break

    return pd.DataFrame(rows)


async def fetch_symbol_history_range(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol: str,
    start_ms: int,
    end_ms: int,
):
    try:
        ratio_tasks = [
            fetch_ratio_history(
                session=session,
                semaphore=semaphore,
                symbol=symbol,
                key=key,
                endpoint=endpoint,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            for key, endpoint in ENDPOINTS.items()
        ]

        oi_task = fetch_open_interest_history(
            session=session,
            semaphore=semaphore,
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
        )

        funding_task = fetch_funding_history(
            session=session,
            semaphore=semaphore,
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
        )

        ratio_dfs = await asyncio.gather(*ratio_tasks)
        oi_df, funding_df = await asyncio.gather(oi_task, funding_task)

        dfs = [df for df in ratio_dfs + [oi_df] if df is not None and not df.empty]

        if len(dfs) < 4:
            return None

        merged = reduce(
            lambda left, right: pd.merge(
                left,
                right,
                on=["symbol", "timestamp_ms"],
                how="inner",
            ),
            dfs,
        )

        if merged.empty:
            return None

        return merged, funding_df

    except Exception as exc:
        print(
            f"[WARN] {symbol} "
            f"{ms_to_utc_str(start_ms)}~{ms_to_utc_str(end_ms)} failed: {exc}"
        )
        return None


# =========================================================
# Spot kline 수집
# =========================================================

async def fetch_spot_klines(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    spot_symbol: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    try:
        url = f"{BASE_SPOT}/api/v3/klines"
        cursor = start_ms
        rows = []

        while cursor <= end_ms:
            params = {
                "symbol": spot_symbol,
                "interval": PERIOD,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            }

            data = await fetch_json(
                session=session,
                url=url,
                params=params,
                semaphore=semaphore,
            )

            if not data:
                break

            for item in data:
                rows.append(
                    {
                        "spot_symbol": spot_symbol,
                        "timestamp_ms": int(item[0]),
                        "spot_close": safe_float(item[4]),
                        "spot_quote_volume_15m": safe_float(item[7]),
                    }
                )

            last_open = int(data[-1][0])
            next_cursor = last_open + PERIOD_MS

            if next_cursor <= cursor:
                break

            cursor = next_cursor

            if len(data) < 1000:
                break

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values(["spot_symbol", "timestamp_ms"])

        bars_24h = int((24 * 60 * 60 * 1000) / PERIOD_MS)

        df["spot_quote_volume_24h"] = (
            df.groupby("spot_symbol")["spot_quote_volume_15m"]
            .rolling(window=bars_24h, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        return df

    except Exception as exc:
        print(f"[WARN] spot {spot_symbol} failed: {exc}")
        return pd.DataFrame()


async def fetch_all_spot_history(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    spot_symbols: list[str],
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    if not spot_symbols:
        return pd.DataFrame()

    tasks = [
        fetch_spot_klines(
            session=session,
            semaphore=semaphore,
            spot_symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        for symbol in spot_symbols
    ]

    out = []

    for idx, task in enumerate(asyncio.as_completed(tasks), 1):
        df = await task

        if df is not None and not df.empty:
            out.append(df)

        if idx % 50 == 0:
            print(f"[SPOT] {idx}/{len(tasks)} done")

    if not out:
        return pd.DataFrame()

    return pd.concat(out, ignore_index=True)


# =========================================================
# Merge
# =========================================================

def merge_funding_asof(
    history_df: pd.DataFrame,
    funding_df: pd.DataFrame,
) -> pd.DataFrame:
    if funding_df.empty:
        history_df["funding_rate"] = np.nan
        history_df["funding_mark_price"] = np.nan
        history_df["funding_time_ms"] = np.nan
        return history_df

    merged_parts = []

    for symbol, left in history_df.groupby("symbol", sort=False):
        right = funding_df[funding_df["symbol"] == symbol].sort_values("funding_time_ms")
        left = left.sort_values("timestamp_ms")

        if right.empty:
            left["funding_rate"] = np.nan
            left["funding_mark_price"] = np.nan
            left["funding_time_ms"] = np.nan
            merged_parts.append(left)
            continue

        tmp = pd.merge_asof(
            left,
            right[["funding_time_ms", "funding_rate", "funding_mark_price"]],
            left_on="timestamp_ms",
            right_on="funding_time_ms",
            direction="backward",
        )

        merged_parts.append(tmp)

    return pd.concat(merged_parts, ignore_index=True)


def merge_spot_history_asof(
    history_df: pd.DataFrame,
    spot_hist_df: pd.DataFrame,
) -> pd.DataFrame:
    if spot_hist_df.empty:
        return history_df

    merged_parts = []

    for spot_symbol, left in history_df.groupby("spot_symbol", dropna=False, sort=False):
        if pd.isna(spot_symbol):
            merged_parts.append(left)
            continue

        right = spot_hist_df[spot_hist_df["spot_symbol"] == spot_symbol].sort_values("timestamp_ms")
        left = left.sort_values("timestamp_ms")

        if right.empty:
            merged_parts.append(left)
            continue

        tmp = pd.merge_asof(
            left,
            right[
                [
                    "timestamp_ms",
                    "spot_quote_volume_15m",
                    "spot_quote_volume_24h",
                    "spot_close",
                ]
            ],
            on="timestamp_ms",
            direction="backward",
            tolerance=PERIOD_MS,
        )

        merged_parts.append(tmp)

    return pd.concat(merged_parts, ignore_index=True)


# =========================================================
# Funding 8시간 환산
# =========================================================

def add_funding_8h_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "funding_rate",
        "funding_time_ms",
        "funding_mark_price",
        "funding_rate_pct",
        "funding_interval_hours",
        "funding_rate_8h",
        "funding_rate_8h_pct",
        "funding_daily_pct",
        "funding_abs_8h_pct",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    if "funding_side" not in out.columns:
        out["funding_side"] = "UNKNOWN"

    if out.empty:
        return out

    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["funding_time_ms"] = pd.to_numeric(out["funding_time_ms"], errors="coerce")
    out["funding_rate"] = pd.to_numeric(out["funding_rate"], errors="coerce")

    events = (
        out[["symbol", "funding_time_ms", "funding_rate"]]
        .dropna(subset=["symbol", "funding_time_ms", "funding_rate"])
        .drop_duplicates(["symbol", "funding_time_ms"], keep="last")
        .copy()
    )

    if events.empty:
        return out

    events["symbol"] = events["symbol"].astype(str).str.upper().str.strip()
    events["funding_time_ms"] = pd.to_numeric(
        events["funding_time_ms"],
        errors="coerce",
    )

    events = events.dropna(subset=["funding_time_ms"]).copy()
    events["funding_time_ms"] = events["funding_time_ms"].round().astype("int64")
    events = events.sort_values(["symbol", "funding_time_ms"])

    events["next_funding_time_ms"] = (
        events.groupby("symbol")["funding_time_ms"].shift(-1)
    )
    events["prev_funding_time_ms"] = (
        events.groupby("symbol")["funding_time_ms"].shift(1)
    )

    next_interval_ms = events["next_funding_time_ms"] - events["funding_time_ms"]
    prev_interval_ms = events["funding_time_ms"] - events["prev_funding_time_ms"]

    interval_ms = next_interval_ms.fillna(prev_interval_ms)
    interval_hours = interval_ms / (60 * 60 * 1000)

    interval_hours = interval_hours.where(
        (interval_hours >= 0.5) & (interval_hours <= 24),
        np.nan,
    )

    events["funding_interval_hours"] = interval_hours

    symbol_median_interval = (
        events.groupby("symbol")["funding_interval_hours"]
        .transform("median")
    )

    events["funding_interval_hours"] = (
        events["funding_interval_hours"]
        .fillna(symbol_median_interval)
        .fillna(8.0)
    )

    events["funding_rate_pct"] = events["funding_rate"] * 100

    events["funding_rate_8h"] = (
        events["funding_rate"] * (8.0 / events["funding_interval_hours"])
    )

    events["funding_rate_8h_pct"] = events["funding_rate_8h"] * 100
    events["funding_daily_pct"] = events["funding_rate_8h_pct"] * 3
    events["funding_abs_8h_pct"] = events["funding_rate_8h_pct"].abs()

    events["funding_side"] = np.select(
        [
            events["funding_rate_8h"] < 0,
            events["funding_rate_8h"] > 0,
        ],
        [
            "LONG_RECEIVES",
            "SHORT_RECEIVES",
        ],
        default="NEUTRAL",
    )

    events["_funding_time_ms_key"] = events["funding_time_ms"].astype("int64")

    feature_map = events[
        [
            "symbol",
            "_funding_time_ms_key",
            "funding_interval_hours",
            "funding_rate_pct",
            "funding_rate_8h",
            "funding_rate_8h_pct",
            "funding_daily_pct",
            "funding_abs_8h_pct",
            "funding_side",
        ]
    ].copy()

    out = out.drop(
        columns=[
            "funding_interval_hours",
            "funding_rate_pct",
            "funding_rate_8h",
            "funding_rate_8h_pct",
            "funding_daily_pct",
            "funding_abs_8h_pct",
            "funding_side",
        ],
        errors="ignore",
    )

    out["_funding_time_ms_key"] = pd.to_numeric(
        out["funding_time_ms"],
        errors="coerce",
    ).round()

    out["_funding_time_ms_key"] = out["_funding_time_ms_key"].astype("Int64")
    feature_map["_funding_time_ms_key"] = feature_map["_funding_time_ms_key"].astype("Int64")

    out = out.merge(
        feature_map,
        on=["symbol", "_funding_time_ms_key"],
        how="left",
    )

    out = out.drop(columns=["_funding_time_ms_key"], errors="ignore")

    return out


# =========================================================
# Scoring
# =========================================================

def add_basic_scoring(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    required = [
        "symbol",
        "timestamp_ms",
        "ls_ratio",
        "ls_acco",
        "ls_position",
        "open_interest",
        "oi_nv",
        "mark_price",
    ]

    missing = [c for c in required if c not in out.columns]

    if missing:
        raise ValueError(f"scoring에 필요한 컬럼이 없음: {missing}")

    numeric_cols = [
        "ls_ratio",
        "ls_acco",
        "ls_position",
        "open_interest",
        "oi_nv",
        "mark_price",
        "spot_quote_volume_24h",
        "spot_quote_volume_24h_current",
        "funding_time_ms",
        "funding_rate",
        "funding_mark_price",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = add_funding_8h_features(out)

    for col in ["ls_ratio", "ls_acco", "ls_position"]:
        out.loc[out[col] <= 0, col] = np.nan

    out["composite_ls"] = 1.0

    for key, weight in WEIGHTS.items():
        out["composite_ls"] *= out[key] ** weight

    out["heat_score"] = np.where(
        out["composite_ls"] < 1,
        -((1 / out["composite_ls"] - 1) * 100),
        (out["composite_ls"] - 1) * 100,
    )

    out["overheat_abs"] = out["heat_score"].abs()

    below = (out[["ls_ratio", "ls_acco", "ls_position"]] < 1).sum(axis=1)
    above = (out[["ls_ratio", "ls_acco", "ls_position"]] > 1).sum(axis=1)

    out["direction"] = np.select(
        [below == 3, above == 3],
        ["SHORT_OVERHEAT", "LONG_OVERHEAT"],
        default="MIXED",
    )

    out["agreement"] = np.select(
        [(below == 3) | (above == 3), (below == 2) | (above == 2)],
        [3, 2],
        default=0,
    )

    out["watch_side"] = np.select(
        [
            out["direction"] == "SHORT_OVERHEAT",
            out["direction"] == "LONG_OVERHEAT",
        ],
        [
            "LONG_WATCH",
            "SHORT_WATCH",
        ],
        default="MIXED_WATCH",
    )

    out["short_skew"] = out["heat_score"].apply(
        lambda x: max(-x, 0) if pd.notna(x) else np.nan
    )

    out["long_skew"] = out["heat_score"].apply(
        lambda x: max(x, 0) if pd.notna(x) else np.nan
    )

    out["short_skew_norm"] = normalize_by_timestamp(out, "short_skew")
    out["long_skew_norm"] = normalize_by_timestamp(out, "long_skew")

    if "spot_market_category" not in out.columns:
        out["spot_market_category"] = "SPOT_MISSING"

    if "spot_quote_volume_24h" not in out.columns:
        out["spot_quote_volume_24h"] = np.nan

    if "spot_quote_volume_24h_current" in out.columns:
        out["spot_quote_volume_24h"] = out["spot_quote_volume_24h"].fillna(
            out["spot_quote_volume_24h_current"]
        )

    out["oi_spot_ratio"] = np.where(
        out["spot_market_category"] == "SPOT_OK",
        out["oi_nv"] / out["spot_quote_volume_24h"],
        np.nan,
    )

    out["oi_pressure_type"] = np.where(
        out["spot_market_category"] == "SPOT_OK",
        "SPOT_RATIO",
        "FUTURES_ONLY",
    )

    out["oi_pressure_raw"] = np.where(
        out["spot_market_category"] == "SPOT_OK",
        out["oi_spot_ratio"],
        out["oi_nv"],
    )

    out["oi_pressure_norm"] = np.nan

    spot_ok = out["spot_market_category"] == "SPOT_OK"
    spot_missing = out["spot_market_category"] == "SPOT_MISSING"

    if spot_ok.any():
        out.loc[spot_ok, "oi_pressure_norm"] = normalize_by_timestamp(
            out.loc[spot_ok].copy(),
            "oi_spot_ratio",
        ).values

    if spot_missing.any():
        out.loc[spot_missing, "oi_pressure_norm"] = normalize_by_timestamp(
            out.loc[spot_missing].copy(),
            "oi_nv",
        ).values

    out["long_lsoi_score"] = 100 * out["short_skew_norm"] * out["oi_pressure_norm"]
    out["short_lsoi_score"] = 100 * out["long_skew_norm"] * out["oi_pressure_norm"]
    out["mixed_lsoi_score"] = out[["long_lsoi_score", "short_lsoi_score"]].max(axis=1)

    out["plot_score"] = out[
        [
            "long_lsoi_score",
            "short_lsoi_score",
            "mixed_lsoi_score",
        ]
    ].max(axis=1)

    return out.replace([np.inf, -np.inf], np.nan)


def classify_quadrant(
    heat_score,
    oi_pressure_norm,
    center_x: float = 20.0,
    center_y: float = 1.0,
) -> str:
    if pd.isna(heat_score) or pd.isna(oi_pressure_norm):
        return "UNKNOWN"

    if abs(heat_score) < center_x and oi_pressure_norm < center_y:
        return "CENTER"

    x = "RIGHT" if heat_score >= 0 else "LEFT"
    y = "UP" if oi_pressure_norm >= center_y else "DOWN"

    return f"{x}_{y}"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "timestamp_ms"]).copy()

    steps_1h = 4
    steps_4h = 16
    steps_24h = 96

    out["quadrant"] = out.apply(
        lambda r: classify_quadrant(r["heat_score"], r["oi_pressure_norm"]),
        axis=1,
    )

    for label, steps in [
        ("1h", steps_1h),
        ("4h", steps_4h),
        ("24h", steps_24h),
    ]:
        out[f"heat_score_prev_{label}"] = out.groupby("symbol")["heat_score"].shift(steps)
        out[f"oi_pressure_norm_prev_{label}"] = out.groupby("symbol")["oi_pressure_norm"].shift(steps)
        out[f"mark_price_prev_{label}"] = out.groupby("symbol")["mark_price"].shift(steps)
        out[f"funding_rate_prev_{label}"] = out.groupby("symbol")["funding_rate"].shift(steps)

        if "funding_rate_8h_pct" in out.columns:
            out[f"funding_rate_8h_pct_prev_{label}"] = (
                out.groupby("symbol")["funding_rate_8h_pct"].shift(steps)
            )
            out[f"d_funding_rate_8h_pct_{label}"] = (
                out["funding_rate_8h_pct"] - out[f"funding_rate_8h_pct_prev_{label}"]
            )
        else:
            out[f"funding_rate_8h_pct_prev_{label}"] = np.nan
            out[f"d_funding_rate_8h_pct_{label}"] = np.nan

        out[f"quadrant_prev_{label}"] = out.groupby("symbol")["quadrant"].shift(steps)

        out[f"dx_{label}"] = out["heat_score"] - out[f"heat_score_prev_{label}"]
        out[f"dy_{label}"] = out["oi_pressure_norm"] - out[f"oi_pressure_norm_prev_{label}"]

        out[f"price_change_{label}_pct"] = (
            (out["mark_price"] / out[f"mark_price_prev_{label}"] - 1) * 100
        )

        out[f"transition_{label}"] = np.where(
            out[f"quadrant_prev_{label}"].notna(),
            out[f"quadrant_prev_{label}"].astype(str) + "_TO_" + out["quadrant"].astype(str),
            np.nan,
        )

    out["timestamp_utc"] = pd.to_datetime(out["timestamp_ms"], unit="ms", utc=True)
    out["timestamp_kst"] = out["timestamp_utc"].dt.tz_convert("Asia/Seoul")

    out["timestamp_utc"] = out["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["timestamp_kst"] = out["timestamp_kst"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return out


# =========================================================
# 저장 / 로드
# =========================================================

def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    return normalize_existing_df(df)


def save_table(df: pd.DataFrame, path: str) -> None:
    ensure_parent_dir(path)

    if path.endswith(".parquet"):
        try:
            df.to_parquet(path, index=False)
            return

        except Exception as exc:
            fallback = path.replace(".parquet", ".csv")
            print(f"[WARN] parquet 저장 실패: {exc}")
            print(f"[WARN] csv로 저장: {fallback}")
            df.to_csv(fallback, index=False, encoding="utf-8-sig")
            return

    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_latest_snapshot(df: pd.DataFrame) -> None:
    if df.empty:
        return

    latest_ts = df["timestamp_ms"].max()
    latest = df[df["timestamp_ms"] == latest_ts].copy()

    cols = [
        "timestamp_utc",
        "timestamp_kst",
        "symbol",
        "watch_side",
        "direction",
        "agreement",
        "ls_ratio",
        "ls_acco",
        "ls_position",
        "composite_ls",
        "heat_score",
        "overheat_abs",
        "open_interest",
        "mark_price",
        "oi_nv",

        "funding_time_ms",
        "funding_rate",
        "funding_rate_pct",
        "funding_interval_hours",
        "funding_rate_8h",
        "funding_rate_8h_pct",
        "funding_daily_pct",
        "funding_abs_8h_pct",
        "funding_side",
        "funding_mark_price",

        "spot_symbol",
        "spot_quote_volume_24h",
        "spot_market_category",
        "oi_spot_ratio",
        "oi_pressure_type",
        "oi_pressure_raw",
        "oi_pressure_norm",
        "short_skew",
        "long_skew",
        "short_skew_norm",
        "long_skew_norm",
        "long_lsoi_score",
        "short_lsoi_score",
        "mixed_lsoi_score",
        "plot_score",
        "quadrant",
        "dx_1h",
        "dy_1h",
        "price_change_1h_pct",
        "funding_rate_8h_pct_prev_1h",
        "d_funding_rate_8h_pct_1h",
        "transition_1h",
        "dx_4h",
        "dy_4h",
        "price_change_4h_pct",
        "funding_rate_8h_pct_prev_4h",
        "d_funding_rate_8h_pct_4h",
        "transition_4h",
        "dx_24h",
        "dy_24h",
        "price_change_24h_pct",
        "funding_rate_8h_pct_prev_24h",
        "d_funding_rate_8h_pct_24h",
        "transition_24h",
    ]

    cols = [c for c in cols if c in latest.columns]

    latest = latest[cols].sort_values(
        ["spot_market_category", "plot_score"],
        ascending=[True, False],
    )

    latest.to_csv("binance_ls_lsoi_score.csv", index=False, encoding="utf-8-sig")

    latest[latest["spot_market_category"] == "SPOT_OK"].to_csv(
        "spot_ok_lsoi.csv",
        index=False,
        encoding="utf-8-sig",
    )

    latest[latest["spot_market_category"] == "SPOT_MISSING"].to_csv(
        "spot_missing_lsoi.csv",
        index=False,
        encoding="utf-8-sig",
    )


def combine_score_and_save(
    existing_df: pd.DataFrame,
    new_raw_df: pd.DataFrame,
    output: str,
    start_keep_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    frames = []

    if existing_df is not None and not existing_df.empty:
        existing_df = normalize_existing_df(existing_df)
        existing_df = existing_df[
            (existing_df["timestamp_ms"] >= start_keep_ms)
            & (existing_df["timestamp_ms"] <= end_ms)
        ].copy()

        if not existing_df.empty:
            frames.append(existing_df)

    if new_raw_df is not None and not new_raw_df.empty:
        new_raw_df = normalize_existing_df(new_raw_df)
        new_raw_df = new_raw_df[
            (new_raw_df["timestamp_ms"] >= start_keep_ms)
            & (new_raw_df["timestamp_ms"] <= end_ms)
        ].copy()

        if not new_raw_df.empty:
            frames.append(new_raw_df)

    if not frames:
        raise RuntimeError("저장할 데이터가 없습니다.")

    common_cols = list(dict.fromkeys([col for frame in frames for col in frame.columns]))

    aligned = []

    for frame in frames:
        frame = frame.copy()

        for col in common_cols:
            if col not in frame.columns:
                frame[col] = np.nan

        aligned.append(frame[common_cols])

    combined_raw = pd.concat(aligned, ignore_index=True)

    combined_raw["symbol"] = combined_raw["symbol"].astype(str).str.upper().str.strip()
    combined_raw["timestamp_ms"] = pd.to_numeric(
        combined_raw["timestamp_ms"],
        errors="coerce",
    )

    combined_raw = combined_raw.dropna(subset=["symbol", "timestamp_ms"]).copy()
    combined_raw["timestamp_ms"] = combined_raw["timestamp_ms"].astype("int64")

    combined_raw = combined_raw.drop_duplicates(
        ["symbol", "timestamp_ms"],
        keep="last",
    )

    combined_raw = combined_raw.sort_values(["symbol", "timestamp_ms"])

    scored = add_basic_scoring(combined_raw)
    scored = add_time_features(scored)

    save_table(scored, output)
    save_latest_snapshot(scored)

    return scored


# =========================================================
# 메인 수집
# =========================================================

async def collect_history(args) -> pd.DataFrame:
    end_ms = floor_to_period_ms(now_ms(), PERIOD_MS)
    start_keep_ms = end_ms - args.days * 24 * 60 * 60 * 1000

    existing_df = pd.DataFrame() if args.force else load_existing(args.output)

    timeout = aiohttp.ClientTimeout(total=30)
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        symbols = await get_usdt_perp_symbols(session)

        print("=== COLLECT CONFIG ===")
        print("mode: ALL USDT PERP")
        print(f"period: {PERIOD} fixed")
        print(f"days: {args.days}")
        print(f"symbols: {len(symbols)}")
        print(f"output: {args.output}")
        print(f"range UTC: {ms_to_utc_str(start_keep_ms)} ~ {ms_to_utc_str(end_ms)}")

        if args.force:
            print("[FORCE] 기존 파일 무시하고 전체 구간 재수집")
            existing_for_skip = pd.DataFrame()
        else:
            existing_for_skip = existing_df.copy()

        fetch_jobs = []

        for symbol in symbols:
            ranges = get_missing_ranges_for_symbol(
                existing_df=existing_for_skip,
                symbol=symbol,
                start_keep_ms=start_keep_ms,
                end_ms=end_ms,
            )

            for start_ms, range_end_ms in ranges:
                fetch_jobs.append((symbol, start_ms, range_end_ms))

        if not fetch_jobs:
            print("[SKIP] 이미 모든 symbol/timestamp 데이터가 존재합니다.")
            return combine_score_and_save(
                existing_df=existing_df,
                new_raw_df=pd.DataFrame(),
                output=args.output,
                start_keep_ms=start_keep_ms,
                end_ms=end_ms,
            )

        fetch_symbols = sorted(set(symbol for symbol, _, _ in fetch_jobs))
        min_fetch_ms = min(start for _, start, _ in fetch_jobs)

        print(f"[FETCH] 새로 받을 심볼: {len(fetch_symbols)}")
        print(f"[FETCH] 새로 받을 구간 수: {len(fetch_jobs)}")
        print(f"[FETCH] 최소 시작 UTC: {ms_to_utc_str(min_fetch_ms)}")

        spot_quote_volume_map = await get_spot_quote_volume_map(session)

        spot_info_df = pd.DataFrame(
            [find_spot_info(s, spot_quote_volume_map) for s in fetch_symbols]
        )

        tasks = [
            fetch_symbol_history_range(
                session=session,
                semaphore=semaphore,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=range_end_ms,
            )
            for symbol, start_ms, range_end_ms in fetch_jobs
        ]

        history_parts = []
        funding_parts = []

        for idx, task in enumerate(asyncio.as_completed(tasks), 1):
            result = await task

            if result is not None:
                hist, fund = result

                if hist is not None and not hist.empty:
                    history_parts.append(hist)

                if fund is not None and not fund.empty:
                    funding_parts.append(fund)

            if idx % 25 == 0:
                print(f"[FUTURES] {idx}/{len(tasks)} ranges done")

        if not history_parts:
            print("[WARN] 새로 수집된 futures history가 없습니다.")
            return combine_score_and_save(
                existing_df=existing_df,
                new_raw_df=pd.DataFrame(),
                output=args.output,
                start_keep_ms=start_keep_ms,
                end_ms=end_ms,
            )

        history_df = pd.concat(history_parts, ignore_index=True)

        funding_df = (
            pd.concat(funding_parts, ignore_index=True)
            if funding_parts
            else pd.DataFrame()
        )

        history_df = merge_funding_asof(history_df, funding_df)
        history_df = history_df.merge(spot_info_df, on="symbol", how="left")

        if not args.no_spot_klines:
            spot_symbols = sorted(
                spot_info_df.loc[
                    spot_info_df["spot_market_category"] == "SPOT_OK",
                    "spot_symbol",
                ]
                .dropna()
                .unique()
                .tolist()
            )

            spot_start_ms = max(0, min_fetch_ms - 24 * 60 * 60 * 1000)

            print(f"[SPOT] kline 대상: {len(spot_symbols)} symbols")
            print(f"[SPOT] range UTC: {ms_to_utc_str(spot_start_ms)} ~ {ms_to_utc_str(end_ms)}")

            spot_hist_df = await fetch_all_spot_history(
                session=session,
                semaphore=semaphore,
                spot_symbols=spot_symbols,
                start_ms=spot_start_ms,
                end_ms=end_ms,
            )

            history_df = merge_spot_history_asof(history_df, spot_hist_df)

        else:
            history_df["spot_quote_volume_24h"] = np.nan

    combined = combine_score_and_save(
        existing_df=existing_df,
        new_raw_df=history_df,
        output=args.output,
        start_keep_ms=start_keep_ms,
        end_ms=end_ms,
    )

    return combined


# =========================================================
# 출력 요약 / CLI
# =========================================================

def print_summary(df: pd.DataFrame, output: str) -> None:
    if df.empty:
        print("데이터 없음")
        return

    latest_ts = df["timestamp_ms"].max()
    latest = df[df["timestamp_ms"] == latest_ts]

    print("\n=== SUMMARY ===")
    print(f"저장 파일: {output}")
    print(f"전체 행: {len(df):,}")
    print(f"심볼 수: {df['symbol'].nunique():,}")
    print(f"최근 시각 UTC: {latest['timestamp_utc'].iloc[0]}")
    print(f"최근 시각 KST: {latest['timestamp_kst'].iloc[0]}")
    print(f"최근 스냅샷: {len(latest):,} symbols")
    print(f"LONG_OVERHEAT: {(latest['direction'] == 'LONG_OVERHEAT').sum():,}")
    print(f"SHORT_OVERHEAT: {(latest['direction'] == 'SHORT_OVERHEAT').sum():,}")
    print(f"MIXED: {(latest['direction'] == 'MIXED').sum():,}")

    expected_bars = len(
        expected_timestamps(
            df["timestamp_ms"].min(),
            df["timestamp_ms"].max(),
        )
    )
    expected_rows_rough = expected_bars * df["symbol"].nunique()

    print("\n=== COVERAGE CHECK ===")
    print(f"봉 개수 범위 기준: {expected_bars:,}")
    print(f"이론상 최대 행 수: {expected_rows_rough:,}")
    print(f"실제 행 수: {len(df):,}")

    if expected_rows_rough > 0:
        print(f"커버리지: {len(df) / expected_rows_rough * 100:.2f}%")

    if "funding_rate_8h_pct" in latest.columns:
        funding_valid = latest["funding_rate_8h_pct"].replace([np.inf, -np.inf], np.nan).dropna()

        print("\n=== FUNDING 8H CHECK ===")

        if funding_valid.empty:
            print("8시간 환산 펀딩 데이터 없음")
        else:
            print(f"펀딩 유효 심볼 수: {len(funding_valid):,}")
            print(f"8h funding min: {funding_valid.min():.4f}%")
            print(f"8h funding median: {funding_valid.median():.4f}%")
            print(f"8h funding max: {funding_valid.max():.4f}%")

            extreme_negative = latest[
                pd.to_numeric(latest["funding_rate_8h_pct"], errors="coerce") <= -0.5
            ].copy()

            extreme_positive = latest[
                pd.to_numeric(latest["funding_rate_8h_pct"], errors="coerce") >= 0.5
            ].copy()

            print(f"8h funding <= -0.5%: {len(extreme_negative):,}")
            print(f"8h funding >= +0.5%: {len(extreme_positive):,}")

            if not extreme_negative.empty:
                print("\n[TOP NEGATIVE FUNDING]")
                cols = [
                    "symbol",
                    "funding_rate_8h_pct",
                    "funding_interval_hours",
                    "funding_side",
                    "heat_score",
                    "oi_pressure_norm",
                    "price_change_4h_pct",
                ]
                cols = [c for c in cols if c in extreme_negative.columns]

                print(
                    extreme_negative[cols]
                    .sort_values("funding_rate_8h_pct")
                    .head(10)
                    .to_string(index=False)
                )

            if not extreme_positive.empty:
                print("\n[TOP POSITIVE FUNDING]")
                cols = [
                    "symbol",
                    "funding_rate_8h_pct",
                    "funding_interval_hours",
                    "funding_side",
                    "heat_score",
                    "oi_pressure_norm",
                    "price_change_4h_pct",
                ]
                cols = [c for c in cols if c in extreme_positive.columns]

                print(
                    extreme_positive[cols]
                    .sort_values("funding_rate_8h_pct", ascending=False)
                    .head(10)
                    .to_string(index=False)
                )

    print("\n최신 CSV도 갱신됨:")
    print("- binance_ls_lsoi_score.csv")
    print("- spot_ok_lsoi.csv")
    print("- spot_missing_lsoi.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Binance ALL USDT perpetual L/S + OI 15m history collector"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="최근 며칠치를 보관/수집할지. 기본 20",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="동시 요청 수. 기본 15",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="history 저장 경로. 기본 data/history_lsoi_15m.csv",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 파일 무시하고 전체 구간 재수집",
    )

    parser.add_argument(
        "--no-spot-klines",
        action="store_true",
        help="spot 15m kline 기반 24h 거래대금 계산 생략",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.days > 29:
        raise ValueError("L/S와 OI Hist 제한 때문에 --days는 29 이하로 두는 게 안전합니다.")

    df = asyncio.run(collect_history(args))
    print_summary(df, args.output)


if __name__ == "__main__":
    main()
