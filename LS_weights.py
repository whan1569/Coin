import asyncio
import aiohttp
import numpy as np
import pandas as pd
from typing import Optional


BASE_FAPI = "https://fapi.binance.com"
BASE_DATA = "https://fapi.binance.com/futures/data"
BASE_SPOT = "https://api.binance.com"

PERIOD = "5m"
LIMIT = 1

CONCURRENCY = 30

# BTC도 포함하려면 빈 set 유지.
# BTC 제외하려면 {"BTCUSDT"}로 변경.
EXCLUDE_SYMBOLS = set()

# LS 내부 가중치
# ls_ratio    = 전체 계정 수 심리
# ls_acco     = 상위 20% 계정 수 심리
# ls_position = 상위 20% 포지션 규모
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


def weighted_geometric_ls(ls_ratio: float, ls_acco: float, ls_position: float) -> float:
    values = {
        "ls_ratio": ls_ratio,
        "ls_acco": ls_acco,
        "ls_position": ls_position,
    }

    composite = 1.0

    for key, weight in WEIGHTS.items():
        value = values[key]

        if value <= 0:
            raise ValueError(f"{key} must be positive. got {value}")

        composite *= value ** weight

    return composite


def ls_heat_score(composite_ls: float) -> float:
    if composite_ls < 1:
        return -((1 / composite_ls - 1) * 100)

    return (composite_ls - 1) * 100


def get_direction(ls_ratio: float, ls_acco: float, ls_position: float) -> str:
    values = [ls_ratio, ls_acco, ls_position]

    if all(v < 1 for v in values):
        return "SHORT_OVERHEAT"

    if all(v > 1 for v in values):
        return "LONG_OVERHEAT"

    return "MIXED"


def get_agreement(ls_ratio: float, ls_acco: float, ls_position: float) -> int:
    values = [ls_ratio, ls_acco, ls_position]

    below = sum(v < 1 for v in values)
    above = sum(v > 1 for v in values)

    if below == 3 or above == 3:
        return 3

    if below == 2 or above == 2:
        return 2

    return 0


def safe_q90(series: pd.Series) -> float:
    value = (
        series.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .quantile(0.90)
    )

    if pd.isna(value) or value <= 0:
        return 1.0

    return float(value)


def normalize_by_q90(series: pd.Series, cap: float = 5.0) -> pd.Series:
    q90 = safe_q90(series)
    return (series / q90).clip(lower=0, upper=cap)


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[dict] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
):
    if semaphore is None:
        async with session.get(url, params=params, timeout=10) as response:
            response.raise_for_status()
            return await response.json()

    async with semaphore:
        async with session.get(url, params=params, timeout=10) as response:
            response.raise_for_status()
            return await response.json()


async def get_usdt_perp_symbols(session: aiohttp.ClientSession) -> list[str]:
    url = f"{BASE_FAPI}/fapi/v1/exchangeInfo"
    data = await fetch_json(session, url)

    symbols = []

    for item in data["symbols"]:
        if item.get("contractType") != "PERPETUAL":
            continue

        if item.get("quoteAsset") != "USDT":
            continue

        if item.get("status") != "TRADING":
            continue

        symbol = item["symbol"]

        if symbol in EXCLUDE_SYMBOLS:
            continue

        symbols.append(symbol)

    return symbols


async def get_mark_price_map(session: aiohttp.ClientSession) -> dict[str, float]:
    url = f"{BASE_FAPI}/fapi/v1/premiumIndex"
    data = await fetch_json(session, url)

    result = {}

    for item in data:
        try:
            result[item["symbol"]] = float(item["markPrice"])
        except Exception:
            continue

    return result


async def get_spot_quote_volume_map(session: aiohttp.ClientSession) -> dict[str, float]:
    url = f"{BASE_SPOT}/api/v3/ticker/24hr"
    data = await fetch_json(session, url)

    result = {}

    for item in data:
        try:
            symbol = str(item["symbol"]).upper().strip()
            quote_volume = float(item["quoteVolume"])

            if quote_volume > 0:
                result[symbol] = quote_volume

        except Exception:
            continue

    return result


def spot_symbol_candidates(futures_symbol: str) -> list[str]:
    candidates = [futures_symbol]

    if futures_symbol.endswith("USDT"):
        base = futures_symbol[:-4]

        numeric_prefixes = [
            "1000000",
            "100000",
            "10000",
            "1000",
            "100",
        ]

        for prefix in numeric_prefixes:
            if base.startswith(prefix):
                candidates.append(base[len(prefix):] + "USDT")

    return list(dict.fromkeys(candidates))


def find_spot_quote_volume_info(
    futures_symbol: str,
    spot_quote_volume_map: dict[str, float],
) -> dict:
    for candidate in spot_symbol_candidates(futures_symbol):
        value = spot_quote_volume_map.get(candidate)

        if value is not None and value > 0:
            return {
                "spot_symbol": candidate,
                "spot_quote_volume_24h": value,
                "spot_market_category": "SPOT_OK",
            }

    return {
        "spot_symbol": None,
        "spot_quote_volume_24h": np.nan,
        "spot_market_category": "SPOT_MISSING",
    }


async def fetch_latest_ratio(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    symbol: str,
) -> Optional[float]:
    url = f"{BASE_DATA}/{endpoint}"
    params = {
        "symbol": symbol,
        "period": PERIOD,
        "limit": LIMIT,
    }

    try:
        data = await fetch_json(session, url, params=params, semaphore=semaphore)

        if not data:
            return None

        return float(data[-1]["longShortRatio"])

    except Exception:
        return None


async def fetch_open_interest(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol: str,
) -> Optional[float]:
    url = f"{BASE_FAPI}/fapi/v1/openInterest"
    params = {"symbol": symbol}

    try:
        data = await fetch_json(session, url, params=params, semaphore=semaphore)
        return float(data["openInterest"])

    except Exception:
        return None


async def fetch_symbol_data(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol: str,
) -> Optional[dict]:
    ratio_tasks = {
        key: fetch_latest_ratio(session, semaphore, endpoint, symbol)
        for key, endpoint in ENDPOINTS.items()
    }

    oi_task = fetch_open_interest(session, semaphore, symbol)

    ratio_results = await asyncio.gather(*ratio_tasks.values())
    open_interest = await oi_task

    if open_interest is None:
        return None

    row = {
        "symbol": symbol,
        "open_interest": open_interest,
    }

    for key, value in zip(ratio_tasks.keys(), ratio_results):
        if value is None:
            return None

        row[key] = value

    return row


def score_ls_row(row: dict) -> dict:
    ls_ratio = row["ls_ratio"]
    ls_acco = row["ls_acco"]
    ls_position = row["ls_position"]

    composite_ls = weighted_geometric_ls(
        ls_ratio=ls_ratio,
        ls_acco=ls_acco,
        ls_position=ls_position,
    )

    heat_score = ls_heat_score(composite_ls)
    direction = get_direction(ls_ratio, ls_acco, ls_position)
    agreement = get_agreement(ls_ratio, ls_acco, ls_position)

    return {
        "symbol": row["symbol"],
        "ls_ratio": ls_ratio,
        "ls_acco": ls_acco,
        "ls_position": ls_position,
        "composite_ls": composite_ls,
        "heat_score": heat_score,
        "overheat_abs": abs(heat_score),
        "direction": direction,
        "agreement": agreement,
        "open_interest": row["open_interest"],
    }


def add_lsoi_scores(
    df: pd.DataFrame,
    mark_price_map: dict[str, float],
    spot_quote_volume_map: dict[str, float],
) -> pd.DataFrame:
    out = df.copy()

    out["mark_price"] = out["symbol"].map(mark_price_map)
    out["oi_nv"] = out["open_interest"] * out["mark_price"]

    out = out.replace([np.inf, -np.inf], np.nan)

    spot_info = out["symbol"].apply(
        lambda symbol: find_spot_quote_volume_info(
            futures_symbol=symbol,
            spot_quote_volume_map=spot_quote_volume_map,
        )
    )

    spot_info_df = pd.DataFrame(spot_info.tolist())

    out = pd.concat(
        [
            out.reset_index(drop=True),
            spot_info_df.reset_index(drop=True),
        ],
        axis=1,
    )

    out = out.replace([np.inf, -np.inf], np.nan)

    out["short_skew"] = out["heat_score"].apply(lambda x: max(-x, 0))
    out["long_skew"] = out["heat_score"].apply(lambda x: max(x, 0))

    out["short_skew_norm"] = normalize_by_q90(out["short_skew"])
    out["long_skew_norm"] = normalize_by_q90(out["long_skew"])

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

    spot_ok_mask = out["spot_market_category"] == "SPOT_OK"
    spot_missing_mask = out["spot_market_category"] == "SPOT_MISSING"

    # 현물 있는 애들은 현물대비 OI로 정규화
    if spot_ok_mask.any():
        out.loc[spot_ok_mask, "oi_pressure_norm"] = normalize_by_q90(
            out.loc[spot_ok_mask, "oi_spot_ratio"]
        )

    # 현물 없는 애들은 선물 OI NV 자체로 정규화
    if spot_missing_mask.any():
        out.loc[spot_missing_mask, "oi_pressure_norm"] = normalize_by_q90(
            out.loc[spot_missing_mask, "oi_nv"]
        )

    out = out.replace([np.inf, -np.inf], np.nan)

    out["long_lsoi_score"] = (
        100
        * out["short_skew_norm"]
        * out["oi_pressure_norm"]
    )

    out["short_lsoi_score"] = (
        100
        * out["long_skew_norm"]
        * out["oi_pressure_norm"]
    )

    out["mixed_lsoi_score"] = out[
        ["long_lsoi_score", "short_lsoi_score"]
    ].max(axis=1)

    out["plot_score"] = out[
        ["long_lsoi_score", "short_lsoi_score", "mixed_lsoi_score"]
    ].max(axis=1)

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

    return out


async def scan_binance_ls() -> pd.DataFrame:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)

    async with aiohttp.ClientSession(connector=connector) as session:
        symbols = await get_usdt_perp_symbols(session)

        mark_price_task = get_mark_price_map(session)
        spot_volume_task = get_spot_quote_volume_map(session)

        tasks = [
            fetch_symbol_data(session, semaphore, symbol)
            for symbol in symbols
        ]

        raw_rows, mark_price_map, spot_quote_volume_map = await asyncio.gather(
            asyncio.gather(*tasks),
            mark_price_task,
            spot_volume_task,
        )

    scored_rows = []

    for row in raw_rows:
        if row is None:
            continue

        try:
            scored_rows.append(score_ls_row(row))
        except Exception:
            continue

    df = pd.DataFrame(scored_rows)

    if df.empty:
        return df

    df = add_lsoi_scores(
        df=df,
        mark_price_map=mark_price_map,
        spot_quote_volume_map=spot_quote_volume_map,
    )

    return df.sort_values(
        ["spot_market_category", "plot_score"],
        ascending=[True, False],
        na_position="last",
    )


def save_outputs(df: pd.DataFrame):
    cols = [
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
    ]

    available_cols = [col for col in cols if col in df.columns]

    all_df = df[available_cols].copy()
    spot_ok_df = df[df["spot_market_category"] == "SPOT_OK"][available_cols].copy()
    spot_missing_df = df[df["spot_market_category"] == "SPOT_MISSING"][available_cols].copy()

    all_df.to_csv(
        "binance_ls_lsoi_score.csv",
        index=False,
        encoding="utf-8-sig",
    )

    spot_ok_df.to_csv(
        "spot_ok_lsoi.csv",
        index=False,
        encoding="utf-8-sig",
    )

    spot_missing_df.to_csv(
        "spot_missing_lsoi.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== SUMMARY ===")
    print(f"ALL: {len(all_df)}")
    print(f"SPOT_OK: {len(spot_ok_df)}")
    print(f"SPOT_MISSING: {len(spot_missing_df)}")
    print("\n저장 완료:")
    print("- binance_ls_lsoi_score.csv")
    print("- spot_ok_lsoi.csv")
    print("- spot_missing_lsoi.csv")


async def main():
    df = await scan_binance_ls()

    if df.empty:
        print("데이터 없음")
        return

    save_outputs(df)


if __name__ == "__main__":
    asyncio.run(main())
