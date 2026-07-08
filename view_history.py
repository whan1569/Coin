# view_history.py

import os
import shlex
import subprocess
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# 설정
# =========================================================

CSV_PATH = "binance_ls_lsoi_score.csv"
HISTORY_CSV_PATH = "data/history_lsoi_15m.csv"
COLLECTOR_PATH = "LS_history_collector.py"


def _tail_text(text: str, max_chars: int = 12000) -> str:
    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    return "... [앞부분 생략] ...\n" + text[-max_chars:]


def run_ls_history_collector(
    collector_path: str,
    days: int,
    concurrency: int,
    extra_args_text: str = "",
) -> dict:
    collector_path = str(collector_path).strip() or COLLECTOR_PATH

    if not os.path.exists(collector_path):
        return {
            "ok": False,
            "returncode": None,
            "cmd": [],
            "stdout": "",
            "stderr": f"수집기 파일을 찾지 못했습니다: {collector_path}",
        }

    cmd = [
        sys.executable,
        collector_path,
        "--days",
        str(int(days)),
        "--concurrency",
        str(int(concurrency)),
    ]

    extra_args_text = str(extra_args_text or "").strip()

    if extra_args_text:
        cmd.extend(shlex.split(extra_args_text, posix=(os.name != "nt")))

    try:
        creationflags = 0

        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = subprocess.CREATE_NO_WINDOW

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )

        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "cmd": cmd,
            "stdout": _tail_text(proc.stdout),
            "stderr": _tail_text(proc.stderr),
        }

    except Exception as e:
        return {
            "ok": False,
            "returncode": None,
            "cmd": cmd,
            "stdout": "",
            "stderr": repr(e),
        }


# =========================================================
# 1. direction / agreement 보조 생성
# =========================================================

def make_direction_from_ls(row) -> str:
    values = [row["ls_ratio"], row["ls_acco"], row["ls_position"]]

    if all(v < 1 for v in values):
        return "SHORT_OVERHEAT"

    if all(v > 1 for v in values):
        return "LONG_OVERHEAT"

    return "MIXED"


def make_agreement_from_ls(row) -> int:
    values = [row["ls_ratio"], row["ls_acco"], row["ls_position"]]

    below = sum(v < 1 for v in values)
    above = sum(v > 1 for v in values)

    if below == 3 or above == 3:
        return 3

    if below == 2 or above == 2:
        return 2

    return 0


def normalize_direction(value) -> str:
    if pd.isna(value):
        return "MIXED"

    v = str(value).strip().upper()

    if v in ["SHORT_OVERHEAT", "SHORT", "SHORT_LS", "SHORT OVERHEAT"]:
        return "SHORT_OVERHEAT"

    if v in ["LONG_OVERHEAT", "LONG", "LONG_LS", "LONG OVERHEAT"]:
        return "LONG_OVERHEAT"

    if v in ["MIXED", "MIXED_LS", "MIX", "MIXED LS"]:
        return "MIXED"

    return "MIXED"


def normalize_spot_market_category(value) -> str:
    if pd.isna(value):
        return "SPOT_MISSING"

    v = str(value).strip().upper()

    if v in ["SPOT_OK", "OK", "SPOT"]:
        return "SPOT_OK"

    if v in ["SPOT_MISSING", "MISSING", "FUTURES_ONLY", "NO_SPOT"]:
        return "SPOT_MISSING"

    return "SPOT_MISSING"


def make_watch_side(row) -> str:
    direction = row["direction"]

    if direction == "SHORT_OVERHEAT":
        return "LONG_WATCH"

    if direction == "LONG_OVERHEAT":
        return "SHORT_WATCH"

    return "MIXED_WATCH"


def make_plot_score(row) -> float:
    direction = row["direction"]

    if direction == "SHORT_OVERHEAT":
        return row.get("long_lsoi_score", 0)

    if direction == "LONG_OVERHEAT":
        return row.get("short_lsoi_score", 0)

    return row.get("mixed_lsoi_score", 0)


# =========================================================
# 1-1. 차트 색/모양 인코딩 보조
# =========================================================

POSITION_COLOR_MAP = {
    "POSITION_UP": "#2D8CFF",       # ls_position > 1: 상승/롱 방향 = 파랑
    "POSITION_DOWN": "#FF5C5C",     # ls_position < 1: 하락/숏 방향 = 빨강
    "POSITION_NEUTRAL": "#A0A0A0",  # ls_position == 1 또는 결측
}

POSITION_SYMBOL_MAP = {
    "ACCO_RATIO_SAME": "circle",          # ACCO, Ratio 둘 다 포지션과 같은 방향
    "ACCO_SAME_ONLY": "triangle-up",      # ACCO만 포지션과 같은 방향
    "RATIO_SAME_ONLY": "triangle-down",   # Ratio만 포지션과 같은 방향
    "BOTH_DIFFERENT": "square",           # ACCO, Ratio 둘 다 포지션과 다른 방향
    "POSITION_NEUTRAL": "diamond",        # 포지션 방향이 중립/결측
}

POSITION_DIRECTION_ORDER = [
    "POSITION_UP",
    "POSITION_DOWN",
    "POSITION_NEUTRAL",
]

POSITION_SYMBOL_ORDER = [
    "ACCO_RATIO_SAME",
    "ACCO_SAME_ONLY",
    "RATIO_SAME_ONLY",
    "BOTH_DIFFERENT",
    "POSITION_NEUTRAL",
]


def ls_value_to_side(value) -> str:
    v = pd.to_numeric(value, errors="coerce")

    if pd.isna(v):
        return "NEUTRAL"

    if v > 1:
        return "UP"

    if v < 1:
        return "DOWN"

    return "NEUTRAL"


def make_position_direction(row) -> str:
    side = ls_value_to_side(row.get("ls_position"))

    if side == "UP":
        return "POSITION_UP"

    if side == "DOWN":
        return "POSITION_DOWN"

    return "POSITION_NEUTRAL"


def make_position_agreement_shape(row) -> str:
    position_side = ls_value_to_side(row.get("ls_position"))
    acco_side = ls_value_to_side(row.get("ls_acco"))
    ratio_side = ls_value_to_side(row.get("ls_ratio"))

    if position_side == "NEUTRAL":
        return "POSITION_NEUTRAL"

    acco_same = acco_side == position_side
    ratio_same = ratio_side == position_side

    if acco_same and ratio_same:
        return "ACCO_RATIO_SAME"

    if acco_same:
        return "ACCO_SAME_ONLY"

    if ratio_same:
        return "RATIO_SAME_ONLY"

    return "BOTH_DIFFERENT"


def add_position_visual_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["position_side"] = out["ls_position"].apply(ls_value_to_side)
    out["acco_side"] = out["ls_acco"].apply(ls_value_to_side)
    out["ratio_side"] = out["ls_ratio"].apply(ls_value_to_side)

    out["position_direction"] = out.apply(make_position_direction, axis=1)
    out["position_agreement_shape"] = out.apply(make_position_agreement_shape, axis=1)

    out["acco_same_as_position"] = (
        (out["position_side"] != "NEUTRAL")
        & (out["acco_side"] == out["position_side"])
    )
    out["ratio_same_as_position"] = (
        (out["position_side"] != "NEUTRAL")
        & (out["ratio_side"] == out["position_side"])
    )

    return out


def apply_position_legend_counts(fig, chart_df: pd.DataFrame):
    """
    Plotly Express가 만든 색+모양 조합 범례 이름 뒤에 현재 표시 개수를 붙인다.
    현재 chart_df 안에 없는 조합은 trace 자체가 없으므로 범례에도 나오지 않는다.
    예: POSITION_DOWN, ACCO_SAME_ONLY (3개)
    """
    required_cols = [
        "position_direction",
        "position_agreement_shape",
    ]

    if any(col not in chart_df.columns for col in required_cols):
        return fig

    combo_counts = (
        chart_df
        .groupby(required_cols, dropna=False)
        .size()
        .to_dict()
    )

    for trace in fig.data:
        raw_name = str(getattr(trace, "name", "") or "")

        # px.scatter(color=A, symbol=B)는 보통 "A, B" 형태의 trace name을 만든다.
        parts = [part.strip() for part in raw_name.split(",", 1)]

        if len(parts) != 2:
            continue

        key = (parts[0], parts[1])
        count = combo_counts.get(key)

        if count is None:
            continue

        trace.name = f"{raw_name} ({int(count)}개)"

    return fig


# =========================================================
# 2. 시계열 시간 처리
# =========================================================

def attach_snapshot_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp_kst" in df.columns:
        df["_snapshot_time"] = pd.to_datetime(df["timestamp_kst"], errors="coerce")

    elif "timestamp_utc" in df.columns:
        ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        df["_snapshot_time"] = ts.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)

    elif "timestamp_ms" in df.columns:
        ts = pd.to_datetime(
            pd.to_numeric(df["timestamp_ms"], errors="coerce"),
            unit="ms",
            errors="coerce",
            utc=True,
        )
        df["_snapshot_time"] = ts.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)

    else:
        raise ValueError(
            "시계열 CSV에 timestamp_kst, timestamp_utc, timestamp_ms 중 하나가 필요합니다."
        )

    df = df.dropna(subset=["_snapshot_time"]).copy()
    df["_snapshot_time"] = df["_snapshot_time"].dt.floor("min")

    return df


@st.cache_data
def load_history_raw(history_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(history_csv_path)
    df = attach_snapshot_time(df)
    return df


def get_recent_history_times(
    history_raw: pd.DataFrame,
    range_mode: str,
) -> list:
    counts = (
        history_raw
        .groupby("_snapshot_time")["symbol"]
        .nunique()
        .sort_index()
    )

    if counts.empty:
        return []

    max_symbol_count = int(counts.max())
    min_required = max(1, int(max_symbol_count * 0.95))

    valid_times = counts[counts >= min_required].index.tolist()

    if not valid_times:
        valid_times = counts.index.tolist()

    valid_times = sorted(valid_times)

    end_time = pd.Timestamp(valid_times[-1])

    if range_mode == "6시간":
        start_time = end_time - pd.Timedelta(hours=6)
    elif range_mode == "24시간":
        start_time = end_time - pd.Timedelta(hours=24)
    elif range_mode == "72시간":
        start_time = end_time - pd.Timedelta(hours=72)
    elif range_mode == "7일":
        start_time = end_time - pd.Timedelta(days=7)
    elif range_mode == "20일":
        start_time = end_time - pd.Timedelta(days=20)
    else:
        start_time = pd.Timestamp(valid_times[0])

    return [t for t in valid_times if pd.Timestamp(t) >= start_time]



def find_nearest_time(target_time, available_times: list):
    if target_time is None or not available_times:
        return None

    target = pd.Timestamp(target_time).floor("min")

    return min(
        available_times,
        key=lambda t: abs(pd.Timestamp(t) - target),
    )


def get_times_between(available_times: list, start_time, end_time) -> list:
    if not available_times:
        return []

    start_ts = pd.Timestamp(start_time)
    end_ts = pd.Timestamp(end_time)

    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts

    return [
        t for t in available_times
        if start_ts <= pd.Timestamp(t) <= end_ts
    ]


def get_preset_start_time(latest_time, earliest_time, range_mode: str):
    latest_ts = pd.Timestamp(latest_time)
    earliest_ts = pd.Timestamp(earliest_time)

    if range_mode == "6시간":
        start_ts = latest_ts - pd.Timedelta(hours=6)
    elif range_mode == "24시간":
        start_ts = latest_ts - pd.Timedelta(hours=24)
    elif range_mode == "72시간":
        start_ts = latest_ts - pd.Timedelta(hours=72)
    elif range_mode == "7일":
        start_ts = latest_ts - pd.Timedelta(days=7)
    elif range_mode == "20일":
        start_ts = latest_ts - pd.Timedelta(days=20)
    else:
        start_ts = earliest_ts

    if start_ts < earliest_ts:
        start_ts = earliest_ts

    return start_ts


def make_range_label(start_time, end_time) -> str:
    if start_time is None or end_time is None:
        return "전체"

    return (
        f"{pd.Timestamp(start_time).strftime('%m-%d %H:%M')}"
        f" ~ {pd.Timestamp(end_time).strftime('%m-%d %H:%M')}"
    )


def sync_text_input_from_time(input_key: str, sync_key: str, target_time):
    if target_time is None:
        return

    target_text = pd.Timestamp(target_time).strftime("%Y-%m-%d %H:%M")

    if st.session_state.get(sync_key) != target_text:
        st.session_state[input_key] = target_text
        st.session_state[sync_key] = target_text


# =========================================================
# 3. 검색 보조
# =========================================================

def resolve_search_symbols(
    search_text: str,
    available_symbols: list[str],
) -> tuple[list[str], list[str]]:
    if not search_text:
        return [], []

    available = sorted(set(str(s).upper().strip() for s in available_symbols))
    available_set = set(available)

    raw_items = search_text.replace(",", " ").split()

    matched_symbols = []
    raw_targets = []

    for item in raw_items:
        token = item.strip().upper()

        if not token:
            continue

        if token.endswith("USDT"):
            target = token
            base_token = token[:-4]
        else:
            target = token + "USDT"
            base_token = token

        raw_targets.append(target)

        if target in available_set:
            matched_symbols.append(target)
            continue

        prefix_matches = [
            symbol for symbol in available
            if symbol.endswith("USDT") and symbol[:-4].startswith(base_token)
        ]

        matched_symbols.extend(prefix_matches)

    matched_symbols = list(dict.fromkeys(matched_symbols))
    raw_targets = list(dict.fromkeys(raw_targets))

    return matched_symbols, raw_targets


def extract_selected_symbols(chart_event) -> list[str]:
    if chart_event is None:
        return []

    selection = None

    if isinstance(chart_event, dict):
        selection = chart_event.get("selection")
    else:
        selection = getattr(chart_event, "selection", None)

    if not selection:
        return []

    if isinstance(selection, dict):
        points = selection.get("points", [])
    else:
        points = getattr(selection, "points", [])

    symbols = []

    for point in points:
        if not isinstance(point, dict):
            continue

        customdata = point.get("customdata")

        if isinstance(customdata, (list, tuple)) and len(customdata) > 0:
            symbol = str(customdata[0]).upper().strip()

            if symbol:
                symbols.append(symbol)

    return list(dict.fromkeys(symbols))


def highlight_active_symbols(row, active_symbols: list[str]):
    symbol = str(row.get("symbol", "")).upper().strip()

    if symbol in active_symbols:
        return [
            "background-color: #ffe066; color: #000000; font-weight: 700;"
            for _ in row
        ]

    return ["" for _ in row]


def get_active_detail_df(
    source_df: pd.DataFrame,
    active_symbols: list[str],
) -> pd.DataFrame:
    if not active_symbols:
        return pd.DataFrame()

    return source_df[source_df["symbol"].isin(active_symbols)].copy()


# =========================================================
# 3-1. 클릭 심볼 시계열 보조
# =========================================================

def add_pressure_timeseries_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "heat_score",
        "oi_pressure_norm",
        "plot_score",
        "mark_price",
        "funding_rate",
        "funding_rate_pct",
        "funding_rate_8h_pct",
        "funding_daily_pct",
        "short_lsoi_score",
        "long_lsoi_score",
        "mixed_lsoi_score",
        "futures_quote_volume_15m",
        "futures_quote_volume_15m_prev",
        "futures_quote_volume_15m_change_pct",
        "futures_quote_volume_15m_change_ratio",
        "spot_quote_volume_15m",
        "spot_quote_volume_15m_prev",
        "spot_quote_volume_15m_change_pct",
        "spot_quote_volume_15m_change_ratio",
        "volume_quote_15m",
        "volume_quote_15m_prev",
        "volume_quote_15m_change_pct",
        "volume_quote_15m_change_ratio",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = add_volume_change_columns(out)

    out["signed_pressure_score"] = out["heat_score"] * out["oi_pressure_norm"]
    out["short_watch_pressure"] = out["heat_score"].clip(lower=0) * out["oi_pressure_norm"]
    out["long_watch_pressure"] = (-out["heat_score"]).clip(lower=0) * out["oi_pressure_norm"]

    if "mark_price" in out.columns:
        def make_price_change_from_start(series: pd.Series) -> pd.Series:
            valid = series.dropna()

            if valid.empty:
                return series * np.nan

            first = valid.iloc[0]

            if pd.isna(first) or first == 0:
                return series * np.nan

            return (series / first - 1) * 100

        out["price_change_from_start_pct"] = (
            out.groupby("symbol")["mark_price"]
            .transform(make_price_change_from_start)
        )
    else:
        out["price_change_from_start_pct"] = np.nan

    return out


def filter_history_window_for_chart(
    df: pd.DataFrame,
    selected_time=None,
    range_mode: str = "전체",
    visible_start_time=None,
    visible_end_time=None,
) -> pd.DataFrame:
    out = df.copy()

    if out.empty:
        return out

    if visible_start_time is not None and visible_end_time is not None:
        start_time = pd.Timestamp(visible_start_time)
        end_time = pd.Timestamp(visible_end_time)

        if start_time > end_time:
            start_time, end_time = end_time, start_time

    else:
        if selected_time is None:
            end_time = pd.Timestamp(out["_snapshot_time"].max())
        else:
            end_time = pd.Timestamp(selected_time)

        if range_mode == "6시간":
            start_time = end_time - pd.Timedelta(hours=6)
        elif range_mode == "24시간":
            start_time = end_time - pd.Timedelta(hours=24)
        elif range_mode == "72시간":
            start_time = end_time - pd.Timedelta(hours=72)
        elif range_mode == "7일":
            start_time = end_time - pd.Timedelta(days=7)
        elif range_mode == "20일":
            start_time = end_time - pd.Timedelta(days=20)
        else:
            start_time = pd.Timestamp(out["_snapshot_time"].min())

    return out[
        (out["_snapshot_time"] >= start_time)
        & (out["_snapshot_time"] <= end_time)
    ].copy()


def get_symbol_pressure_history(
    history_csv_path: str,
    active_symbols: list[str],
    selected_time=None,
    range_mode: str = "전체",
    visible_start_time=None,
    visible_end_time=None,
) -> pd.DataFrame:
    if not active_symbols:
        return pd.DataFrame()

    raw = load_history_raw(history_csv_path)

    symbols = [
        str(symbol).upper().strip()
        for symbol in active_symbols
        if str(symbol).strip()
    ]

    if not symbols:
        return pd.DataFrame()

    raw["symbol"] = raw["symbol"].astype(str).str.upper().str.strip()

    symbol_df = raw[raw["symbol"].isin(symbols)].copy()

    if symbol_df.empty:
        return pd.DataFrame()

    symbol_df = filter_history_window_for_chart(
        symbol_df,
        selected_time=selected_time,
        range_mode=range_mode,
        visible_start_time=visible_start_time,
        visible_end_time=visible_end_time,
    )

    if symbol_df.empty:
        return pd.DataFrame()

    symbol_df = preprocess_lsoi_df(
        symbol_df,
        source_name=f"{history_csv_path} pressure history",
    )

    symbol_df = add_pressure_timeseries_columns(symbol_df)
    symbol_df = symbol_df.sort_values(["symbol", "_snapshot_time"])

    return symbol_df


def make_pressure_hover_data(df: pd.DataFrame) -> dict:
    candidates = {
        "_snapshot_time_plot": True,
        "symbol": True,
        "direction": True,
        "watch_side": True,
        "heat_score": ":.2f",
        "oi_pressure_norm": ":.3f",
        "signed_pressure_score": ":.2f",
        "short_watch_pressure": ":.2f",
        "long_watch_pressure": ":.2f",
        "plot_score": ":.2f",
        "mark_price": ":.8f",
        "price_change_from_start_pct": ":.2f",
        "volume_quote_15m": ":.2f",
        "volume_quote_15m_prev": ":.2f",
        "volume_quote_15m_change_pct": ":.2f",
        "futures_quote_volume_15m": ":.2f",
        "futures_quote_volume_15m_change_pct": ":.2f",
        "spot_quote_volume_15m": ":.2f",
        "spot_quote_volume_15m_change_pct": ":.2f",
        "funding_rate": ":.6f",
        "funding_rate_pct": ":.4f",
        "funding_rate_8h_pct": ":.4f",
        "funding_daily_pct": ":.4f",
        "funding_side": True,
    }

    return make_hover_data(df, candidates)


def add_selected_time_shape(fig, selected_snapshot_time):
    if selected_snapshot_time is None:
        return fig

    selected_ts = pd.Timestamp(selected_snapshot_time).floor("min")
    selected_x = selected_ts.strftime("%Y-%m-%d %H:%M:%S")

    # Plotly의 add_vline(annotation_*)는 datetime/Timestamp 축에서
    # 내부적으로 x0+x1 평균을 계산하다가 TypeError가 날 수 있다.
    # 그래서 vline 헬퍼 대신 shape + annotation을 분리해서 넣는다.
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=selected_x,
        x1=selected_x,
        y0=0,
        y1=1,
        line=dict(
            color="yellow",
            width=2,
            dash="dash",
        ),
        layer="above",
    )

    fig.add_annotation(
        x=selected_x,
        y=1,
        xref="x",
        yref="paper",
        text="선택 시점",
        showarrow=False,
        yanchor="bottom",
        font=dict(color="yellow"),
        bgcolor="rgba(0,0,0,0.35)",
    )

    fig.update_xaxes(
        tickformat="%m-%d %H:%M",
        hoverformat="%Y-%m-%d %H:%M",
    )

    return fig




# =========================================================
# 3-2. 거래량 변화율 보조
# =========================================================

def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _sort_for_symbol_time(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["symbol"]

    if "timestamp_ms" in df.columns:
        sort_cols.append("timestamp_ms")
    elif "_snapshot_time" in df.columns:
        sort_cols.append("_snapshot_time")

    return df.sort_values(sort_cols).copy()


def add_volume_change_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    volume_numeric_cols = [
        "futures_base_volume_15m",
        "futures_quote_volume_15m",
        "futures_quote_volume_15m_prev",
        "futures_quote_volume_15m_change_pct",
        "futures_quote_volume_15m_change_ratio",
        "spot_quote_volume_15m",
        "spot_quote_volume_15m_prev",
        "spot_quote_volume_15m_change_pct",
        "spot_quote_volume_15m_change_ratio",
        "volume_quote_15m",
        "volume_quote_15m_prev",
        "volume_quote_15m_change_pct",
        "volume_quote_15m_change_ratio",
        "futures_trade_count_15m",
    ]

    for col in volume_numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "volume_quote_15m" not in out.columns:
        out["volume_quote_15m"] = np.nan

        source_col = _first_existing_col(
            out,
            [
                "futures_quote_volume_15m",
                "spot_quote_volume_15m",
            ],
        )

        if source_col is not None:
            out["volume_quote_15m"] = out[source_col]
    else:
        fallback_col = _first_existing_col(
            out,
            [
                "futures_quote_volume_15m",
                "spot_quote_volume_15m",
            ],
        )

        if fallback_col is not None:
            out["volume_quote_15m"] = out["volume_quote_15m"].fillna(out[fallback_col])

    can_shift = (
        "symbol" in out.columns
        and (
            "timestamp_ms" in out.columns
            or "_snapshot_time" in out.columns
        )
    )

    if can_shift and "volume_quote_15m" in out.columns:
        sorted_out = _sort_for_symbol_time(out)
        shifted = sorted_out.groupby("symbol")["volume_quote_15m"].shift(1)
        out.loc[sorted_out.index, "_volume_quote_15m_prev_calc"] = shifted.values

        if "volume_quote_15m_prev" not in out.columns:
            out["volume_quote_15m_prev"] = out["_volume_quote_15m_prev_calc"]
        else:
            out["volume_quote_15m_prev"] = out["volume_quote_15m_prev"].fillna(
                out["_volume_quote_15m_prev_calc"]
            )

        out = out.drop(columns=["_volume_quote_15m_prev_calc"], errors="ignore")

    if "volume_quote_15m_prev" not in out.columns:
        out["volume_quote_15m_prev"] = np.nan

    valid_prev = out["volume_quote_15m_prev"].notna() & (out["volume_quote_15m_prev"] > 0)

    if "volume_quote_15m_change_ratio" not in out.columns:
        out["volume_quote_15m_change_ratio"] = np.nan

    out.loc[valid_prev, "volume_quote_15m_change_ratio"] = (
        out.loc[valid_prev, "volume_quote_15m"]
        / out.loc[valid_prev, "volume_quote_15m_prev"]
    )

    if "volume_quote_15m_change_pct" not in out.columns:
        out["volume_quote_15m_change_pct"] = np.nan

    out.loc[valid_prev, "volume_quote_15m_change_pct"] = (
        out.loc[valid_prev, "volume_quote_15m_change_ratio"] - 1
    ) * 100

    for prefix in ["futures", "spot"]:
        current_col = f"{prefix}_quote_volume_15m"

        if current_col not in out.columns:
            continue

        prev_col = f"{prefix}_quote_volume_15m_prev"
        ratio_col = f"{prefix}_quote_volume_15m_change_ratio"
        pct_col = f"{prefix}_quote_volume_15m_change_pct"

        if can_shift:
            sorted_out = _sort_for_symbol_time(out)
            shifted = sorted_out.groupby("symbol")[current_col].shift(1)
            calc_col = f"_{prev_col}_calc"
            out.loc[sorted_out.index, calc_col] = shifted.values

            if prev_col not in out.columns:
                out[prev_col] = out[calc_col]
            else:
                out[prev_col] = out[prev_col].fillna(out[calc_col])

            out = out.drop(columns=[calc_col], errors="ignore")

        if prev_col not in out.columns:
            out[prev_col] = np.nan

        valid_prev = out[prev_col].notna() & (out[prev_col] > 0)

        if ratio_col not in out.columns:
            out[ratio_col] = np.nan

        out.loc[valid_prev, ratio_col] = (
            out.loc[valid_prev, current_col]
            / out.loc[valid_prev, prev_col]
        )

        if pct_col not in out.columns:
            out[pct_col] = np.nan

        out.loc[valid_prev, pct_col] = (out.loc[valid_prev, ratio_col] - 1) * 100

    return out


def make_hover_data(df: pd.DataFrame, candidates: dict) -> dict:
    return {
        key: value
        for key, value in candidates.items()
        if key in df.columns
    }


# =========================================================
# 4. 기준 CSV 전처리 / 로드
# =========================================================

def preprocess_lsoi_df(df: pd.DataFrame, source_name: str = "dataframe") -> pd.DataFrame:
    required_cols = [
        "symbol",
        "ls_ratio",
        "ls_acco",
        "ls_position",
        "composite_ls",
        "heat_score",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{source_name}에 필요한 컬럼이 없음: {missing}")

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    numeric_cols = [
        "ls_ratio",
        "ls_acco",
        "ls_position",
        "composite_ls",
        "heat_score",
        "agreement",
        "overheat_abs",
        "open_interest",
        "mark_price",
        "oi_nv",
        "spot_quote_volume_24h",
        "oi_spot_ratio",
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
        "funding_time_ms",
        "funding_rate",
        "funding_rate_pct",
        "funding_interval_hours",
        "funding_rate_8h",
        "funding_rate_8h_pct",
        "funding_daily_pct",
        "funding_abs_8h_pct",
        "price_change_1h_pct",
        "price_change_4h_pct",
        "price_change_24h_pct",
        "futures_base_volume_15m",
        "futures_quote_volume_15m",
        "futures_quote_volume_15m_prev",
        "futures_quote_volume_15m_change_pct",
        "futures_quote_volume_15m_change_ratio",
        "spot_quote_volume_15m",
        "spot_quote_volume_15m_prev",
        "spot_quote_volume_15m_change_pct",
        "spot_quote_volume_15m_change_ratio",
        "volume_quote_15m",
        "volume_quote_15m_prev",
        "volume_quote_15m_change_pct",
        "volume_quote_15m_change_ratio",
        "futures_trade_count_15m",
        "dx_1h",
        "dy_1h",
        "dx_4h",
        "dy_4h",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "ls_ratio",
            "ls_acco",
            "ls_position",
            "composite_ls",
            "heat_score",
        ]
    ).copy()

    df = df.replace([np.inf, -np.inf], np.nan)

    if "direction" in df.columns:
        df["direction"] = df["direction"].apply(normalize_direction)
    else:
        df["direction"] = df.apply(make_direction_from_ls, axis=1)

    if "agreement" in df.columns:
        df["agreement"] = df["agreement"].fillna(0).astype(int)
    else:
        df["agreement"] = df.apply(make_agreement_from_ls, axis=1)

    if "overheat_abs" not in df.columns:
        df["overheat_abs"] = df["heat_score"].abs()
    else:
        df["overheat_abs"] = df["overheat_abs"].fillna(df["heat_score"].abs())

    if "spot_market_category" in df.columns:
        df["spot_market_category"] = df["spot_market_category"].apply(
            normalize_spot_market_category
        )
    else:
        if "spot_quote_volume_24h" in df.columns:
            df["spot_market_category"] = np.where(
                df["spot_quote_volume_24h"].notna()
                & (df["spot_quote_volume_24h"] > 0),
                "SPOT_OK",
                "SPOT_MISSING",
            )
        else:
            df["spot_market_category"] = "SPOT_MISSING"

    if "spot_symbol" not in df.columns:
        df["spot_symbol"] = np.where(
            df["spot_market_category"] == "SPOT_OK",
            df["symbol"],
            None,
        )

    if "watch_side" not in df.columns:
        df["watch_side"] = df.apply(make_watch_side, axis=1)

    if "short_skew" not in df.columns:
        df["short_skew"] = df["heat_score"].apply(lambda x: max(-x, 0))

    if "long_skew" not in df.columns:
        df["long_skew"] = df["heat_score"].apply(lambda x: max(x, 0))

    if "oi_pressure_type" not in df.columns:
        df["oi_pressure_type"] = np.where(
            df["spot_market_category"] == "SPOT_OK",
            "SPOT_RATIO",
            "FUTURES_ONLY",
        )

    if "oi_spot_ratio" not in df.columns:
        if "oi_nv" in df.columns and "spot_quote_volume_24h" in df.columns:
            df["oi_spot_ratio"] = np.where(
                df["spot_market_category"] == "SPOT_OK",
                df["oi_nv"] / df["spot_quote_volume_24h"],
                np.nan,
            )
        else:
            df["oi_spot_ratio"] = np.nan

    if "oi_pressure_raw" not in df.columns:
        df["oi_pressure_raw"] = np.where(
            df["spot_market_category"] == "SPOT_OK",
            df["oi_spot_ratio"],
            df["oi_nv"] if "oi_nv" in df.columns else np.nan,
        )

    if "oi_pressure_norm" not in df.columns:
        df["oi_pressure_norm"] = df["oi_pressure_raw"]

    if "plot_score" not in df.columns:
        df["plot_score"] = df.apply(make_plot_score, axis=1)

    df["plot_score"] = pd.to_numeric(
        df["plot_score"],
        errors="coerce",
    ).fillna(0)

    df = add_volume_change_columns(df)

    return df


@st.cache_data
def load_lsoi_score(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return preprocess_lsoi_df(df, csv_path)


@st.cache_data
def load_history_snapshot(
    history_csv_path: str,
    snapshot_time,
) -> pd.DataFrame:
    raw = load_history_raw(history_csv_path)

    selected_time = pd.Timestamp(snapshot_time).floor("min")

    snap = raw[raw["_snapshot_time"] == selected_time].copy()

    if snap.empty:
        available_times = sorted(
            pd.to_datetime(raw["_snapshot_time"].dropna().unique()).tolist()
        )

        if not available_times:
            raise ValueError("시계열 CSV에 사용 가능한 시간이 없음")

        nearest_time = min(
            available_times,
            key=lambda t: abs(pd.Timestamp(t) - selected_time),
        )

        snap = raw[raw["_snapshot_time"] == pd.Timestamp(nearest_time)].copy()

    return preprocess_lsoi_df(snap, f"{history_csv_path} @ {selected_time}")


# =========================================================
# 5. 필터 / 정렬
# =========================================================

def apply_market_category_filter(
    df: pd.DataFrame,
    market_category: str,
) -> pd.DataFrame:
    if market_category == "ALL":
        return df.copy()

    return df[df["spot_market_category"] == market_category].copy()


def apply_direction_filter(
    df: pd.DataFrame,
    direction_filter: str,
) -> pd.DataFrame:
    if direction_filter == "ALL":
        return df.copy()

    return df[df["direction"] == direction_filter].copy()


def apply_rank(
    df: pd.DataFrame,
    direction_filter: str,
    rank_mode: str,
) -> pd.DataFrame:
    if rank_mode == "LS*OI 점수":
        if direction_filter == "SHORT_OVERHEAT":
            return df.sort_values(
                ["long_lsoi_score", "heat_score"],
                ascending=[False, True],
                na_position="last",
            )

        if direction_filter == "LONG_OVERHEAT":
            return df.sort_values(
                ["short_lsoi_score", "heat_score"],
                ascending=[False, False],
                na_position="last",
            )

        if direction_filter == "MIXED":
            return df.sort_values(
                ["mixed_lsoi_score", "overheat_abs"],
                ascending=[False, False],
                na_position="last",
            )

        return df.sort_values(
            ["plot_score", "overheat_abs"],
            ascending=[False, False],
            na_position="last",
        )

    if rank_mode == "OI 압력":
        return df.sort_values(
            ["oi_pressure_norm", "plot_score"],
            ascending=[False, False],
            na_position="last",
        )

    if rank_mode == "현물대비 OI":
        return df.sort_values(
            ["oi_spot_ratio", "plot_score"],
            ascending=[False, False],
            na_position="last",
        )

    if rank_mode == "Heat 상위":
        return df.sort_values(
            "heat_score",
            ascending=False,
            na_position="last",
        )

    if rank_mode == "Heat 하위":
        return df.sort_values(
            "heat_score",
            ascending=True,
            na_position="last",
        )

    return df.sort_values(
        ["plot_score", "overheat_abs"],
        ascending=[False, False],
        na_position="last",
    )


def add_searched_rows_to_view(
    view: pd.DataFrame,
    result: pd.DataFrame,
    search_symbols: list[str],
) -> pd.DataFrame:
    if not search_symbols:
        return view

    searched_rows = result[result["symbol"].isin(search_symbols)].copy()

    if searched_rows.empty:
        return view

    out = pd.concat([view, searched_rows], ignore_index=True)
    out = out.drop_duplicates(subset=["symbol"], keep="last")

    return out


# =========================================================
# 6. Streamlit 화면
# =========================================================

def main():
    st.set_page_config(
        page_title="Binance LS × OI Viewer",
        layout="wide",
    )

    st.title("Binance LS × OI Pressure Viewer")

    st.caption(
        "기준 데이터는 binance_ls_lsoi_score.csv 또는 "
        "data/history_lsoi_15m.csv의 선택 시간 스냅샷을 사용합니다. "
        "시장 카테고리는 ALL / SPOT_OK / SPOT_MISSING 3개로 나누고, "
        "LS 방향은 별도 필터로 봅니다. "
        "차트 크기와 OI 압력 값은 CSV 값을 그대로 사용합니다."
    )

    # -----------------------------------------------------
    # 사이드바
    # -----------------------------------------------------
    with st.sidebar:
        st.header("보기 설정")

        market_category = st.selectbox(
            "시장 카테고리",
            [
                "ALL",
                "SPOT_OK",
                "SPOT_MISSING",
            ],
            index=0,
        )

        direction_filter = st.selectbox(
            "LS 방향",
            [
                "ALL",
                "SHORT_OVERHEAT",
                "LONG_OVERHEAT",
                "MIXED",
            ],
            index=0,
        )

        rank_mode = st.selectbox(
            "정렬 기준",
            [
                "LS*OI 점수",
                "OI 압력",
                "현물대비 OI",
                "Heat 상위",
                "Heat 하위",
            ],
            index=0,
        )

        display_count = st.number_input(
            "표시 개수",
            min_value=1,
            max_value=1000,
            value=50,
            step=10,
        )

        st.divider()

        search_text = st.text_input(
            "심볼 검색",
            value="",
            placeholder="예: XMR 또는 XMRUSDT, LIT, HUMA, vv",
        )

        label_only_search = st.checkbox(
            "검색한 심볼만 이름 표시",
            value=True,
        )

        show_all_symbol = st.checkbox(
            "전체 심볼명 표시",
            value=False,
        )

        log_y = st.checkbox(
            "Y축 로그 스케일",
            value=True,
        )

        st.divider()
        st.header("시간 선택")

        default_use_history = os.path.exists(HISTORY_CSV_PATH)

        use_history = st.checkbox(
            "시계열 데이터 사용",
            value=default_use_history,
        )

        history_csv_path = HISTORY_CSV_PATH
        selected_snapshot_time = None
        visible_start_time = None
        visible_end_time = None
        visible_range_label = "현재 스냅샷"
        history_range = "전체"
        all_snapshot_times = []
        visible_snapshot_times = []

        if use_history:
            history_csv_path = st.text_input(
                "시계열 CSV 경로",
                value=HISTORY_CSV_PATH,
            )

            try:
                history_raw = load_history_raw(history_csv_path)
                all_snapshot_times = get_recent_history_times(history_raw, "전체")

                if not all_snapshot_times:
                    st.warning("시계열 CSV에 선택 가능한 시간이 없습니다.")
                    use_history = False
                else:
                    earliest_snapshot_time = all_snapshot_times[0]
                    latest_snapshot_time = all_snapshot_times[-1]

                    st.subheader("표현 범위")

                    visible_range_preset = st.selectbox(
                        "표현 범위 프리셋",
                        [
                            "6시간",
                            "24시간",
                            "72시간",
                            "7일",
                            "20일",
                            "전체",
                        ],
                        index=4,
                        help="프리셋은 Min/Max 슬라이더 값을 잡아주는 용도입니다. 실제 표현 범위는 아래 Min/Max가 결정합니다.",
                    )

                    range_key = "visible_time_range_slider"
                    force_latest_snapshot = st.session_state.pop(
                        "force_latest_snapshot",
                        False,
                    )

                    preset_start_time = find_nearest_time(
                        get_preset_start_time(
                            latest_snapshot_time,
                            earliest_snapshot_time,
                            visible_range_preset,
                        ),
                        all_snapshot_times,
                    )
                    preset_end_time = latest_snapshot_time

                    stored_range_value = st.session_state.get(range_key)

                    if (
                        force_latest_snapshot
                        or range_key not in st.session_state
                        or not isinstance(stored_range_value, tuple)
                        or len(stored_range_value) != 2
                        or stored_range_value[0] not in all_snapshot_times
                        or stored_range_value[1] not in all_snapshot_times
                    ):
                        st.session_state[range_key] = (
                            preset_start_time,
                            preset_end_time,
                        )

                    if st.button("표현 범위 프리셋 적용", use_container_width=True):
                        st.session_state[range_key] = (
                            preset_start_time,
                            preset_end_time,
                        )
                        st.rerun()

                    visible_range_value = st.select_slider(
                        "표현 범위 Min / Max",
                        options=all_snapshot_times,
                        key=range_key,
                        format_func=lambda x: pd.Timestamp(x).strftime("%m-%d %H:%M"),
                    )

                    if not isinstance(visible_range_value, tuple):
                        visible_range_value = (
                            visible_range_value,
                            visible_range_value,
                        )

                    visible_start_time, visible_end_time = visible_range_value

                    if pd.Timestamp(visible_start_time) > pd.Timestamp(visible_end_time):
                        visible_start_time, visible_end_time = (
                            visible_end_time,
                            visible_start_time,
                        )

                    visible_snapshot_times = get_times_between(
                        all_snapshot_times,
                        visible_start_time,
                        visible_end_time,
                    )

                    if not visible_snapshot_times:
                        visible_snapshot_times = all_snapshot_times
                        visible_start_time = all_snapshot_times[0]
                        visible_end_time = all_snapshot_times[-1]
                        st.session_state[range_key] = (
                            visible_start_time,
                            visible_end_time,
                        )

                    visible_range_label = make_range_label(
                        visible_start_time,
                        visible_end_time,
                    )
                    history_range = visible_range_label

                    st.caption(
                        "표현 범위: "
                        f"{pd.Timestamp(visible_start_time).strftime('%Y-%m-%d %H:%M')}"
                        " ~ "
                        f"{pd.Timestamp(visible_end_time).strftime('%Y-%m-%d %H:%M')} KST"
                    )

                    st.subheader("선택 시점")

                    cursor_key = "selected_cursor_time"

                    if (
                        force_latest_snapshot
                        or cursor_key not in st.session_state
                        or find_nearest_time(
                            st.session_state.get(cursor_key),
                            all_snapshot_times,
                        ) is None
                    ):
                        st.session_state[cursor_key] = latest_snapshot_time

                    current_cursor_time = find_nearest_time(
                        st.session_state[cursor_key],
                        all_snapshot_times,
                    )

                    if current_cursor_time not in visible_snapshot_times:
                        current_cursor_time = find_nearest_time(
                            current_cursor_time,
                            visible_snapshot_times,
                        )
                        st.session_state[cursor_key] = current_cursor_time

                    sync_text_input_from_time(
                        input_key="cursor_input_text",
                        sync_key="cursor_input_sync_target",
                        target_time=current_cursor_time,
                    )

                    cursor_input_text = st.text_input(
                        "선택시점 직접 입력",
                        key="cursor_input_text",
                        placeholder="예: 2026-07-05 05:00",
                    )

                    input_apply_col, latest_col = st.columns(2)

                    with input_apply_col:
                        if st.button("선택시점 적용", use_container_width=True):
                            parsed_time = pd.to_datetime(
                                cursor_input_text,
                                errors="coerce",
                            )

                            if pd.isna(parsed_time):
                                st.warning("시간 형식이 올바르지 않습니다. 예: 2026-07-05 05:00")
                            else:
                                nearest_time = find_nearest_time(
                                    parsed_time,
                                    all_snapshot_times,
                                )

                                if nearest_time is not None:
                                    st.session_state[cursor_key] = nearest_time

                                    if nearest_time not in visible_snapshot_times:
                                        st.session_state[range_key] = (
                                            min(visible_start_time, nearest_time),
                                            max(visible_end_time, nearest_time),
                                        )

                                    st.rerun()

                    with latest_col:
                        if st.button("최신 시점", use_container_width=True):
                            st.session_state[cursor_key] = latest_snapshot_time

                            if latest_snapshot_time not in visible_snapshot_times:
                                st.session_state[range_key] = (
                                    visible_start_time,
                                    latest_snapshot_time,
                                )

                            st.rerun()

                    current_cursor_time = find_nearest_time(
                        st.session_state[cursor_key],
                        visible_snapshot_times,
                    )

                    if current_cursor_time is None:
                        current_cursor_time = visible_snapshot_times[-1]
                        st.session_state[cursor_key] = current_cursor_time

                    current_cursor_index = visible_snapshot_times.index(
                        current_cursor_time,
                    )

                    prev_col, next_col = st.columns(2)

                    with prev_col:
                        if st.button("◀ 이전 15분", use_container_width=True):
                            prev_index = max(0, current_cursor_index - 1)
                            st.session_state[cursor_key] = visible_snapshot_times[prev_index]
                            st.rerun()

                    with next_col:
                        if st.button("다음 15분 ▶", use_container_width=True):
                            next_index = min(
                                len(visible_snapshot_times) - 1,
                                current_cursor_index + 1,
                            )
                            st.session_state[cursor_key] = visible_snapshot_times[next_index]
                            st.rerun()

                    selected_by_slider = st.select_slider(
                        "선택시점 슬라이더",
                        options=visible_snapshot_times,
                        value=current_cursor_time,
                        format_func=lambda x: pd.Timestamp(x).strftime("%m-%d %H:%M"),
                    )

                    if selected_by_slider != current_cursor_time:
                        st.session_state[cursor_key] = selected_by_slider
                        st.rerun()

                    selected_snapshot_time = st.session_state[cursor_key]

                    st.caption(
                        "선택 시점: "
                        f"{pd.Timestamp(selected_snapshot_time).strftime('%Y-%m-%d %H:%M')} KST"
                    )

            except Exception as e:
                st.warning(f"시계열 데이터 로드 실패: {e}")
                use_history = False
                selected_snapshot_time = None
                visible_start_time = None
                visible_end_time = None
        st.divider()

        st.header("클릭 차트")
        st.caption("하단 시계열 차트는 위의 표현 범위 Min/Max를 그대로 사용합니다.")

        auto_show_top_symbols = st.checkbox(
            "선택/검색 없을 때 상위 심볼 자동 표시",
            value=True,
        )

        auto_top_symbol_count = st.number_input(
            "자동 표시 심볼 수",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
        )

        if st.button("선택 심볼 초기화", use_container_width=True):
            st.session_state["active_symbols"] = []
            st.rerun()

        st.divider()
        st.header("데이터 갱신")

        collector_path = st.text_input(
            "수집기 경로",
            value=COLLECTOR_PATH,
        )

        collector_days = st.number_input(
            "수집 일수",
            min_value=1,
            max_value=365,
            value=20,
            step=1,
        )

        collector_concurrency = st.number_input(
            "수집 동시성",
            min_value=1,
            max_value=30,
            value=3,
            step=1,
            help="403 Forbidden이 뜨면 1~3으로 낮추는 편이 안전합니다.",
        )

        collector_extra_args = st.text_input(
            "추가 실행 옵션",
            value="",
            placeholder="예: --mode all",
        )

        if st.button("🔄 LS 수집 후 새로고침", use_container_width=True):
            with st.spinner("LS_history_collector.py 실행 중..."):
                collector_result = run_ls_history_collector(
                    collector_path=collector_path,
                    days=int(collector_days),
                    concurrency=int(collector_concurrency),
                    extra_args_text=collector_extra_args,
                )

            st.session_state["collector_last_result"] = collector_result

            if collector_result.get("ok"):
                st.cache_data.clear()
                st.session_state["force_latest_snapshot"] = True
                st.session_state["last_refresh_time"] = pd.Timestamp.now(
                    tz="Asia/Seoul",
                ).strftime("%Y-%m-%d %H:%M:%S")
                st.rerun()

        if st.button("↻ 화면만 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.session_state["force_latest_snapshot"] = True
            st.session_state["last_refresh_time"] = pd.Timestamp.now(
                tz="Asia/Seoul",
            ).strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

        if "last_refresh_time" in st.session_state:
            st.caption(f"마지막 새로고침: {st.session_state['last_refresh_time']} KST")

        collector_last_result = st.session_state.get("collector_last_result")

        if collector_last_result:
            if collector_last_result.get("ok"):
                st.success("최근 수집기 실행 성공")
            else:
                st.error("최근 수집기 실행 실패")

            with st.expander("최근 수집기 로그"):
                cmd = collector_last_result.get("cmd") or []

                if cmd:
                    st.code(" ".join(cmd), language="bash")

                stdout = collector_last_result.get("stdout") or ""
                stderr = collector_last_result.get("stderr") or ""

                if stdout:
                    st.text_area(
                        "stdout",
                        value=stdout,
                        height=180,
                    )

                if stderr:
                    st.text_area(
                        "stderr",
                        value=stderr,
                        height=120,
                    )

    # -----------------------------------------------------
    # 데이터 로드
    # -----------------------------------------------------
    try:
        if use_history and selected_snapshot_time is not None:
            result = load_history_snapshot(
                history_csv_path,
                selected_snapshot_time,
            )
        else:
            result = load_lsoi_score(CSV_PATH)

    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return

    search_symbols, raw_search_targets = resolve_search_symbols(
        search_text,
        result["symbol"].tolist(),
    )

    # -----------------------------------------------------
    # 필터 적용
    # -----------------------------------------------------
    filtered_df = apply_market_category_filter(result, market_category)
    filtered_df = apply_direction_filter(filtered_df, direction_filter)

    sorted_df = apply_rank(filtered_df, direction_filter, rank_mode)
    view = sorted_df.head(int(display_count)).copy()

    view = add_searched_rows_to_view(view, result, search_symbols)

    # -----------------------------------------------------
    # 검색 결과 안내
    # -----------------------------------------------------
    if search_text:
        found_symbols = sorted(set(result["symbol"]) & set(search_symbols))
        missing_targets = sorted(set(raw_search_targets) - set(found_symbols))

        msg = f"검색어: {search_text}"

        if raw_search_targets:
            msg += f" / 해석: {', '.join(raw_search_targets)}"

        if found_symbols:
            msg += f" / CSV 발견: {', '.join(found_symbols)}"

        if missing_targets and not found_symbols:
            msg += f" / CSV에 없음: {', '.join(missing_targets)}"

        st.info(msg)

    # -----------------------------------------------------
    # 현재 선택 데이터 안내
    # -----------------------------------------------------
    if use_history and selected_snapshot_time is not None:
        st.info(
            "시계열 스냅샷 사용 중: "
            f"{pd.Timestamp(selected_snapshot_time).strftime('%Y-%m-%d %H:%M')} KST"
            f" / 표현 범위: {visible_range_label}"
        )
    else:
        st.info("현재 스냅샷 CSV 사용 중: binance_ls_lsoi_score.csv")

    # -----------------------------------------------------
    # 상단 요약
    # -----------------------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("ALL", len(result))
    col2.metric("SPOT_OK", len(result[result["spot_market_category"] == "SPOT_OK"]))
    col3.metric(
        "SPOT_MISSING",
        len(result[result["spot_market_category"] == "SPOT_MISSING"]),
    )

    col4, col5, col6, col7 = st.columns(4)

    col4.metric("SHORT", len(result[result["direction"] == "SHORT_OVERHEAT"]))
    col5.metric("LONG", len(result[result["direction"] == "LONG_OVERHEAT"]))
    col6.metric("MIXED", len(result[result["direction"] == "MIXED"]))
    col7.metric("현재 표시", len(view))

    if view.empty:
        st.warning("조건에 맞는 데이터가 없음")
        return

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("시장 카테고리", market_category)
    c2.metric("LS 방향", direction_filter)
    c3.metric("정렬 기준", rank_mode)
    c4.metric("최고 LS*OI 점수", f"{view['plot_score'].max():.2f}")

    c5, c6, c7, c8 = st.columns(4)

    c5.metric("최고 heat_score", f"{view['heat_score'].max():.2f}")
    c6.metric("최저 heat_score", f"{view['heat_score'].min():.2f}")
    c7.metric("최고 OI 압력", f"{view['oi_pressure_norm'].max():.3f}")
    c8.metric("평균 과열 절댓값", f"{view['overheat_abs'].mean():.2f}")

    c9, c10, c11, c12 = st.columns(4)

    if "volume_quote_15m_change_pct" in view.columns:
        volume_change_valid = pd.to_numeric(
            view["volume_quote_15m_change_pct"],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        c9.metric(
            "최고 15m 거래량 변화율",
            "-" if volume_change_valid.empty else f"{volume_change_valid.max():.2f}%",
        )
    else:
        c9.metric("최고 15m 거래량 변화율", "-")

    if "volume_quote_15m" in view.columns:
        volume_valid = pd.to_numeric(
            view["volume_quote_15m"],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        c10.metric(
            "최고 15m 거래대금",
            "-" if volume_valid.empty else f"{volume_valid.max():,.0f}",
        )
    else:
        c10.metric("최고 15m 거래대금", "-")

    funding_metric_col = (
        "funding_rate_8h_pct"
        if "funding_rate_8h_pct" in view.columns
        else "funding_rate"
        if "funding_rate" in view.columns
        else None
    )

    if funding_metric_col is not None:
        funding_valid = pd.to_numeric(
            view[funding_metric_col],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        c11.metric(
            "최고 펀딩",
            "-" if funding_valid.empty else f"{funding_valid.max():.4f}",
        )
        c12.metric(
            "최저 펀딩",
            "-" if funding_valid.empty else f"{funding_valid.min():.4f}",
        )
    else:
        c11.metric("최고 펀딩", "-")
        c12.metric("최저 펀딩", "-")

    # -----------------------------------------------------
    # 차트
    # -----------------------------------------------------
    chart_df = view.copy()

    if log_y:
        chart_df = chart_df[chart_df["oi_pressure_norm"] > 0].copy()

    if chart_df.empty:
        st.warning("차트에 표시할 데이터가 없음")
        return

    chart_df["label_symbol"] = ""

    if show_all_symbol:
        chart_df["label_symbol"] = chart_df["symbol"]

    elif label_only_search and search_symbols:
        chart_df.loc[
            chart_df["symbol"].isin(search_symbols),
            "label_symbol",
        ] = chart_df["symbol"]

    text_col = "label_symbol"

    chart_df["plot_size"] = pd.to_numeric(
        chart_df["plot_score"],
        errors="coerce",
    ).fillna(0)

    if chart_df["plot_size"].max() <= 0:
        chart_df["plot_size"] = 1

    # 색: ls_position 방향
    #   POSITION_UP   = 파랑
    #   POSITION_DOWN = 빨강
    # 모양: ls_position 방향과 ls_acco / ls_ratio가 얼마나 같은지
    #   ACCO_RATIO_SAME = 동그라미
    #   ACCO_SAME_ONLY  = 삼각형
    #   RATIO_SAME_ONLY = 역삼각형
    #   BOTH_DIFFERENT  = 네모
    chart_df = add_position_visual_columns(chart_df)

    title_time = ""

    if use_history and selected_snapshot_time is not None:
        title_time = (
            f" | {pd.Timestamp(selected_snapshot_time).strftime('%Y-%m-%d %H:%M')} KST"
        )

    scatter_hover_candidates = {
        "symbol": True,
        "position_direction": True,
        "position_agreement_shape": True,
        "position_side": True,
        "acco_side": True,
        "ratio_side": True,
        "acco_same_as_position": True,
        "ratio_same_as_position": True,
        "watch_side": True,
        "direction": True,
        "spot_market_category": True,
        "spot_symbol": True,
        "oi_pressure_type": True,
        "agreement": True,
        "ls_ratio": ":.3f",
        "ls_acco": ":.3f",
        "ls_position": ":.3f",
        "composite_ls": ":.3f",
        "heat_score": ":.2f",
        "overheat_abs": ":.2f",
        "open_interest": ":.4f",
        "mark_price": ":.8f",
        "oi_nv": ":.2f",
        "spot_quote_volume_24h": ":.2f",
        "volume_quote_15m": ":.2f",
        "volume_quote_15m_prev": ":.2f",
        "volume_quote_15m_change_pct": ":.2f",
        "futures_quote_volume_15m": ":.2f",
        "futures_quote_volume_15m_change_pct": ":.2f",
        "spot_quote_volume_15m": ":.2f",
        "spot_quote_volume_15m_change_pct": ":.2f",
        "oi_spot_ratio": ":.5f",
        "oi_pressure_raw": ":.5f",
        "oi_pressure_norm": ":.3f",
        "short_skew": ":.3f",
        "long_skew": ":.3f",
        "short_skew_norm": ":.3f",
        "long_skew_norm": ":.3f",
        "long_lsoi_score": ":.2f",
        "short_lsoi_score": ":.2f",
        "mixed_lsoi_score": ":.2f",
        "plot_score": ":.2f",
        "funding_rate": ":.6f",
        "funding_rate_8h_pct": ":.4f",
        "funding_side": True,
        "label_symbol": False,
        "plot_size": False,
    }

    fig = px.scatter(
        chart_df,
        x="heat_score",
        y="oi_pressure_norm",
        text=text_col,
        size="plot_size",
        size_max=45,
        color="position_direction",
        symbol="position_agreement_shape",
        color_discrete_map=POSITION_COLOR_MAP,
        symbol_map=POSITION_SYMBOL_MAP,
        category_orders={
            "position_direction": POSITION_DIRECTION_ORDER,
            "position_agreement_shape": POSITION_SYMBOL_ORDER,
        },
        custom_data=["symbol"],
        hover_data=make_hover_data(chart_df, scatter_hover_candidates),
        title=(
            f"{market_category} | {direction_filter} | "
            f"{rank_mode} | {int(display_count)}개"
            f" | 색=Position 방향 / 모양=ACCO·Ratio 일치"
            f"{title_time}"
        ),
        labels={
            "heat_score": "Heat score / LS direction",
            "oi_pressure_norm": "OI Pressure Norm",
            "plot_size": "LS × OI Pressure score",
            "position_direction": "Position Direction Color",
            "position_agreement_shape": "ACCO / Ratio Agreement Shape",
            "direction": "LS Direction",
            "spot_market_category": "Market Category",
        },
        log_y=log_y,
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        annotation_text="Heat 0",
        annotation_position="top",
    )

    fig.add_hline(
        y=1,
        line_dash="dash",
        annotation_text="OI Pressure Q90 기준",
        annotation_position="bottom right",
    )

    fig.update_traces(
        textposition="top center",
        marker=dict(opacity=0.72),
    )

    fig = apply_position_legend_counts(fig, chart_df)

    if search_symbols:
        search_chart_df = chart_df[chart_df["symbol"].isin(search_symbols)].copy()

        if not search_chart_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=search_chart_df["heat_score"],
                    y=search_chart_df["oi_pressure_norm"],
                    mode="markers+text",
                    text=search_chart_df["symbol"],
                    customdata=search_chart_df[["symbol"]].values,
                    textposition="top center",
                    marker=dict(
                        size=22,
                        symbol="circle-open",
                        line=dict(width=4),
                    ),
                    name="SEARCHED",
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Heat: %{x:.2f}<br>"
                        "OI Pressure Norm: %{y:.3f}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=True,
                )
            )

    fig.update_layout(height=760)

    # -----------------------------------------------------
    # 차트 표시 + 클릭/선택 이벤트
    # -----------------------------------------------------
    chart_event = None

    try:
        chart_event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
        )
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

    clicked_symbols = extract_selected_symbols(chart_event)

    if "active_symbols" not in st.session_state:
        st.session_state["active_symbols"] = []

    if clicked_symbols:
        st.session_state["active_symbols"] = list(
            dict.fromkeys(clicked_symbols)
        )

    active_symbols = list(
        dict.fromkeys(
            search_symbols
            + clicked_symbols
            + st.session_state.get("active_symbols", [])
        )
    )

    if active_symbols:
        timeseries_symbols = active_symbols
        timeseries_source_label = "검색 / 클릭"
    elif use_history and auto_show_top_symbols:
        timeseries_symbols = (
            view["symbol"]
            .astype(str)
            .str.upper()
            .str.strip()
            .head(int(auto_top_symbol_count))
            .tolist()
        )
        timeseries_symbols = list(dict.fromkeys(timeseries_symbols))
        timeseries_source_label = f"현재 상위 {len(timeseries_symbols)}개"
    else:
        timeseries_symbols = []
        timeseries_source_label = "검색 / 클릭"

    # -----------------------------------------------------
    # 검색 / 클릭 심볼 상세
    # -----------------------------------------------------
    if active_symbols:
        active_detail_df = get_active_detail_df(result, active_symbols)

        if not active_detail_df.empty:
            st.subheader("검색 / 클릭 심볼 상세")

            detail_cols = [
                "symbol",
                "watch_side",
                "direction",
                "spot_market_category",
                "spot_symbol",
                "oi_pressure_type",
                "heat_score",
                "overheat_abs",
                "oi_pressure_norm",
                "plot_score",
                "long_lsoi_score",
                "short_lsoi_score",
                "mixed_lsoi_score",
                "ls_ratio",
                "ls_acco",
                "ls_position",
                "composite_ls",
                "open_interest",
                "mark_price",
                "oi_nv",
                "spot_quote_volume_24h",
                "oi_spot_ratio",
                "oi_pressure_raw",
                "volume_quote_15m",
                "volume_quote_15m_prev",
                "volume_quote_15m_change_pct",
                "futures_quote_volume_15m",
                "futures_quote_volume_15m_change_pct",
                "spot_quote_volume_15m",
                "spot_quote_volume_15m_change_pct",
            ]

            extra_cols = [
                "funding_rate",
                "funding_rate_pct",
                "funding_rate_8h_pct",
                "funding_daily_pct",
                "funding_side",
                "price_change_1h_pct",
                "price_change_4h_pct",
                "dx_1h",
                "dy_1h",
                "dx_4h",
                "dy_4h",
                "transition_1h",
                "transition_4h",
            ]

            detail_cols += extra_cols
            detail_cols = [col for col in detail_cols if col in active_detail_df.columns]

            st.dataframe(
                active_detail_df[detail_cols]
                .style.apply(
                    lambda row: highlight_active_symbols(row, active_symbols),
                    axis=1,
                ),
                use_container_width=True,
                height=180,
            )

    # -----------------------------------------------------
    # 클릭 / 검색 / 자동 상위 심볼 압력 + 가격 시계열 차트
    # -----------------------------------------------------
    if timeseries_symbols and use_history:
        pressure_range = visible_range_label

        pressure_history_df = get_symbol_pressure_history(
            history_csv_path=history_csv_path,
            active_symbols=timeseries_symbols,
            selected_time=selected_snapshot_time,
            range_mode="전체",
            visible_start_time=visible_start_time,
            visible_end_time=visible_end_time,
        )

        if not pressure_history_df.empty:
            st.subheader(f"{timeseries_source_label} 심볼 압력 시계열")

            metric_options = {
                "Signed 압력": "signed_pressure_score",
                "숏 후보 압력": "short_watch_pressure",
                "롱 후보 압력": "long_watch_pressure",
                "Plot score": "plot_score",
                "가격 변화율": "price_change_from_start_pct",
                "15m 거래량 변화율": "volume_quote_15m_change_pct",
                "8h 환산 펀딩비": "funding_rate_8h_pct",
                "Funding rate": "funding_rate",
            }

            available_metric_options = {
                label: col
                for label, col in metric_options.items()
                if col in pressure_history_df.columns
            }

            selected_pressure_metric_label = st.selectbox(
                "압력 시계열 지표",
                list(available_metric_options.keys()),
                index=0,
                key="pressure_metric_select",
            )

            selected_pressure_metric = available_metric_options[
                selected_pressure_metric_label
            ]

            pressure_history_df["_snapshot_time"] = pd.to_datetime(
                pressure_history_df["_snapshot_time"],
                errors="coerce",
            )

            pressure_history_df["_snapshot_time_plot"] = (
                pressure_history_df["_snapshot_time"]
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            )

            pressure_fig = px.line(
                pressure_history_df,
                x="_snapshot_time",
                y=selected_pressure_metric,
                color="symbol",
                markers=True,
                hover_data=make_pressure_hover_data(pressure_history_df),
                title=(
                    f"{', '.join(timeseries_symbols)} | "
                    f"{selected_pressure_metric_label} | {pressure_range}"
                ),
                labels={
                    "_snapshot_time": "Time KST",
                    selected_pressure_metric: selected_pressure_metric_label,
                    "symbol": "Symbol",
                },
            )

            if selected_pressure_metric in [
                "signed_pressure_score",
                "short_watch_pressure",
                "long_watch_pressure",
                "price_change_from_start_pct",
                "funding_rate",
                "funding_rate_8h_pct",
                "volume_quote_15m_change_pct",
            ]:
                pressure_fig.add_hline(
                    y=0,
                    line_dash="dash",
                    annotation_text="0",
                    annotation_position="bottom right",
                )

            pressure_fig = add_selected_time_shape(
                pressure_fig,
                selected_snapshot_time,
            )

            pressure_fig.update_layout(height=430)

            st.plotly_chart(
                pressure_fig,
                use_container_width=True,
            )

            # -------------------------------------------------
            # 실제 가격(mark_price) 라인 차트
            # -------------------------------------------------
            st.subheader(f"{timeseries_source_label} 심볼 실제 가격 라인")

            price_metric_options = {
                "실제 가격": "mark_price",
                "시작 대비 가격 변화율": "price_change_from_start_pct",
            }

            available_price_metric_options = {
                label: col
                for label, col in price_metric_options.items()
                if col in pressure_history_df.columns
            }

            if available_price_metric_options:
                selected_price_metric_label = st.selectbox(
                    "가격 차트 지표",
                    list(available_price_metric_options.keys()),
                    index=0,
                    key="price_metric_select",
                )

                selected_price_metric = available_price_metric_options[
                    selected_price_metric_label
                ]

                price_fig = px.line(
                    pressure_history_df,
                    x="_snapshot_time",
                    y=selected_price_metric,
                    color="symbol",
                    markers=True,
                    hover_data=make_pressure_hover_data(pressure_history_df),
                    title=(
                        f"{', '.join(timeseries_symbols)} | "
                        f"{selected_price_metric_label} | {pressure_range}"
                    ),
                    labels={
                        "_snapshot_time": "Time KST",
                        selected_price_metric: selected_price_metric_label,
                        "symbol": "Symbol",
                    },
                )

                if selected_price_metric == "price_change_from_start_pct":
                    price_fig.add_hline(
                        y=0,
                        line_dash="dash",
                        annotation_text="0%",
                        annotation_position="bottom right",
                    )

                price_fig = add_selected_time_shape(
                    price_fig,
                    selected_snapshot_time,
                )

                price_fig.update_layout(height=380)

                st.plotly_chart(
                    price_fig,
                    use_container_width=True,
                )


            # -------------------------------------------------
            # 직전 15분 대비 거래량 변화율 라인 차트
            # -------------------------------------------------
            st.subheader(f"{timeseries_source_label} 심볼 15분 거래량 변화율")

            volume_metric_options = {
                "직전 15분 대비 거래량 변화율": "volume_quote_15m_change_pct",
                "15분 거래대금": "volume_quote_15m",
                "직전 15분 거래대금": "volume_quote_15m_prev",
                "선물 15분 거래대금": "futures_quote_volume_15m",
                "현물 15분 거래대금": "spot_quote_volume_15m",
            }

            available_volume_metric_options = {
                label: col
                for label, col in volume_metric_options.items()
                if col in pressure_history_df.columns
                and pd.to_numeric(
                    pressure_history_df[col],
                    errors="coerce",
                ).replace([np.inf, -np.inf], np.nan).notna().any()
            }

            if available_volume_metric_options:
                selected_volume_metric_label = st.selectbox(
                    "거래량 차트 지표",
                    list(available_volume_metric_options.keys()),
                    index=0,
                    key="volume_metric_select",
                )

                selected_volume_metric = available_volume_metric_options[
                    selected_volume_metric_label
                ]

                volume_fig = px.line(
                    pressure_history_df,
                    x="_snapshot_time",
                    y=selected_volume_metric,
                    color="symbol",
                    markers=True,
                    hover_data=make_pressure_hover_data(pressure_history_df),
                    title=(
                        f"{', '.join(timeseries_symbols)} | "
                        f"{selected_volume_metric_label} | {pressure_range}"
                    ),
                    labels={
                        "_snapshot_time": "Time KST",
                        selected_volume_metric: selected_volume_metric_label,
                        "symbol": "Symbol",
                    },
                )

                if selected_volume_metric.endswith("_change_pct"):
                    volume_fig.add_hline(
                        y=0,
                        line_dash="dash",
                        annotation_text="0%",
                        annotation_position="bottom right",
                    )

                if selected_volume_metric.endswith("_change_ratio"):
                    volume_fig.add_hline(
                        y=1,
                        line_dash="dash",
                        annotation_text="1x",
                        annotation_position="bottom right",
                    )

                volume_fig = add_selected_time_shape(
                    volume_fig,
                    selected_snapshot_time,
                )

                volume_fig.update_layout(height=380)

                st.plotly_chart(
                    volume_fig,
                    use_container_width=True,
                )
            else:
                st.info(
                    "거래량 변화율 컬럼이 없습니다. "
                    "수집기를 새 버전으로 실행해 volume_quote_15m_change_pct를 생성하세요."
                )

            # -------------------------------------------------
            # 펀딩비 라인 차트
            # -------------------------------------------------
            st.subheader(f"{timeseries_source_label} 심볼 펀딩비 라인")

            funding_metric_options = {
                "8시간 환산 펀딩비(%)": "funding_rate_8h_pct",
                "원본 펀딩비(%)": "funding_rate_pct",
                "원본 펀딩비": "funding_rate",
                "일 환산 펀딩비(%)": "funding_daily_pct",
            }

            available_funding_metric_options = {
                label: col
                for label, col in funding_metric_options.items()
                if col in pressure_history_df.columns
                and pd.to_numeric(
                    pressure_history_df[col],
                    errors="coerce",
                ).replace([np.inf, -np.inf], np.nan).notna().any()
            }

            if available_funding_metric_options:
                selected_funding_metric_label = st.selectbox(
                    "펀딩비 차트 지표",
                    list(available_funding_metric_options.keys()),
                    index=0,
                    key="funding_metric_select",
                )

                selected_funding_metric = available_funding_metric_options[
                    selected_funding_metric_label
                ]

                funding_fig = px.line(
                    pressure_history_df,
                    x="_snapshot_time",
                    y=selected_funding_metric,
                    color="symbol",
                    markers=True,
                    hover_data=make_pressure_hover_data(pressure_history_df),
                    title=(
                        f"{', '.join(timeseries_symbols)} | "
                        f"{selected_funding_metric_label} | {pressure_range}"
                    ),
                    labels={
                        "_snapshot_time": "Time KST",
                        selected_funding_metric: selected_funding_metric_label,
                        "symbol": "Symbol",
                    },
                )

                funding_fig.add_hline(
                    y=0,
                    line_dash="dash",
                    annotation_text="0",
                    annotation_position="bottom right",
                )

                funding_fig = add_selected_time_shape(
                    funding_fig,
                    selected_snapshot_time,
                )

                funding_fig.update_layout(height=380)

                st.plotly_chart(
                    funding_fig,
                    use_container_width=True,
                )
            else:
                st.info("펀딩비 컬럼이 없어 펀딩비 차트를 표시하지 못했습니다.")


            pressure_table_cols = [
                "_snapshot_time",
                "symbol",
                "direction",
                "watch_side",
                "heat_score",
                "oi_pressure_norm",
                "signed_pressure_score",
                "short_watch_pressure",
                "long_watch_pressure",
                "plot_score",
                "mark_price",
                "price_change_from_start_pct",
                "volume_quote_15m",
                "volume_quote_15m_prev",
                "volume_quote_15m_change_pct",
                "futures_quote_volume_15m",
                "futures_quote_volume_15m_change_pct",
                "spot_quote_volume_15m",
                "spot_quote_volume_15m_change_pct",
                "funding_rate",
                "funding_rate_pct",
                "funding_rate_8h_pct",
                "funding_daily_pct",
                "funding_side",
                "transition_1h",
                "transition_4h",
            ]

            pressure_table_cols = [
                col for col in pressure_table_cols
                if col in pressure_history_df.columns
            ]

            with st.expander("선택 심볼 시계열 원본 보기"):
                st.dataframe(
                    pressure_history_df[pressure_table_cols].tail(300),
                    use_container_width=True,
                    height=320,
                )

        else:
            st.info("표시 대상 심볼의 시계열 데이터를 찾지 못했습니다.")

    # -----------------------------------------------------
    # 테이블
    # -----------------------------------------------------
    st.subheader("필터 결과")

    view_cols = [
        "symbol",
        "watch_side",
        "direction",
        "spot_market_category",
        "spot_symbol",
        "oi_pressure_type",
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
        "spot_quote_volume_24h",
        "oi_spot_ratio",
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
        "volume_quote_15m",
        "volume_quote_15m_prev",
        "volume_quote_15m_change_pct",
        "volume_quote_15m_change_ratio",
        "futures_quote_volume_15m",
        "futures_quote_volume_15m_prev",
        "futures_quote_volume_15m_change_pct",
        "spot_quote_volume_15m",
        "spot_quote_volume_15m_prev",
        "spot_quote_volume_15m_change_pct",
        "funding_rate",
        "funding_rate_pct",
        "funding_rate_8h_pct",
        "funding_daily_pct",
        "funding_side",
        "price_change_1h_pct",
        "price_change_4h_pct",
        "dx_1h",
        "dy_1h",
        "dx_4h",
        "dy_4h",
        "transition_1h",
        "transition_4h",
    ]

    available_cols = [col for col in view_cols if col in view.columns]

    if active_symbols:
        st.dataframe(
            view[available_cols]
            .style.apply(
                lambda row: highlight_active_symbols(row, active_symbols),
                axis=1,
            ),
            use_container_width=True,
            height=460,
        )
    else:
        st.dataframe(
            view[available_cols],
            use_container_width=True,
            height=460,
        )

    download_time = ""

    if use_history and selected_snapshot_time is not None:
        download_time = "_" + pd.Timestamp(selected_snapshot_time).strftime("%Y%m%d_%H%M")

    st.download_button(
        label="현재 결과 CSV 다운로드",
        data=view[available_cols].to_csv(index=False, encoding="utf-8-sig"),
        file_name=(
            f"{market_category}_{direction_filter}_"
            f"{rank_mode}_{int(display_count)}{download_time}.csv"
        ),
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
