# view_history.py

import os
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
        "short_lsoi_score",
        "long_lsoi_score",
        "mixed_lsoi_score",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

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
    selected_time,
    range_mode: str,
) -> pd.DataFrame:
    out = df.copy()

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
    selected_time,
    range_mode: str,
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
        "funding_rate": ":.6f",
    }

    return {
        key: value
        for key, value in candidates.items()
        if key in df.columns
    }


def add_selected_time_shape(fig, selected_snapshot_time):
    if selected_snapshot_time is None:
        return fig

    selected_x = pd.Timestamp(selected_snapshot_time).strftime("%Y-%m-%d %H:%M:%S")

    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=selected_x,
        x1=selected_x,
        y0=0,
        y1=1,
        line=dict(
            dash="dash",
            width=1,
        ),
    )

    fig.add_annotation(
        x=selected_x,
        y=1,
        xref="x",
        yref="paper",
        text="선택 시점",
        showarrow=False,
        yanchor="bottom",
    )

    return fig


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
        "funding_rate",
        "price_change_1h_pct",
        "price_change_4h_pct",
        "price_change_24h_pct",
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
        history_range = "20일"

        if use_history:
            history_csv_path = st.text_input(
                "시계열 CSV 경로",
                value=HISTORY_CSV_PATH,
            )

            history_range = st.selectbox(
                "시간 범위",
                [
                    "6시간",
                    "24시간",
                    "72시간",
                    "7일",
                    "20일",
                    "전체",
                ],
                index=4,
            )

            try:
                history_raw = load_history_raw(history_csv_path)
                snapshot_times = get_recent_history_times(history_raw, history_range)

                if not snapshot_times:
                    st.warning("시계열 CSV에 선택 가능한 시간이 없습니다.")
                    use_history = False
                else:
                    selected_snapshot_time = st.select_slider(
                        "시간 선택",
                        options=snapshot_times,
                        value=snapshot_times[-1],
                        format_func=lambda x: pd.Timestamp(x).strftime("%m-%d %H:%M"),
                    )

                    st.caption(
                        f"선택 시점: "
                        f"{pd.Timestamp(selected_snapshot_time).strftime('%Y-%m-%d %H:%M')} KST"
                    )

            except Exception as e:
                st.warning(f"시계열 데이터 로드 실패: {e}")
                use_history = False
                selected_snapshot_time = None

        st.divider()

        st.header("클릭 차트")
        pressure_chart_default_range = st.selectbox(
            "클릭 심볼 시계열 범위",
            [
                "6시간",
                "24시간",
                "72시간",
                "7일",
                "20일",
                "전체",
                "현재 시간 범위와 동일",
            ],
            index=4,
        )

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

        if st.button("선택 심볼 초기화"):
            st.session_state["active_symbols"] = []
            st.rerun()

        if st.button("새로고침"):
            st.cache_data.clear()
            st.rerun()

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

    title_time = ""

    if use_history and selected_snapshot_time is not None:
        title_time = (
            f" | {pd.Timestamp(selected_snapshot_time).strftime('%Y-%m-%d %H:%M')} KST"
        )

    fig = px.scatter(
        chart_df,
        x="heat_score",
        y="oi_pressure_norm",
        text=text_col,
        size="plot_size",
        size_max=45,
        color="direction",
        symbol="spot_market_category",
        custom_data=["symbol"],
        hover_data={
            "symbol": True,
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
            "label_symbol": False,
            "plot_size": False,
        },
        title=(
            f"{market_category} | {direction_filter} | "
            f"{rank_mode} | {int(display_count)}개"
            f"{title_time}"
        ),
        labels={
            "heat_score": "Heat score / LS direction",
            "oi_pressure_norm": "OI Pressure Norm",
            "plot_size": "LS × OI Pressure score",
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
            ]

            extra_cols = [
                "funding_rate",
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
        pressure_range = (
            history_range
            if pressure_chart_default_range == "현재 시간 범위와 동일"
            else pressure_chart_default_range
        )

        pressure_history_df = get_symbol_pressure_history(
            history_csv_path=history_csv_path,
            active_symbols=timeseries_symbols,
            selected_time=selected_snapshot_time,
            range_mode=pressure_range,
        )

        if not pressure_history_df.empty:
            st.subheader(f"{timeseries_source_label} 심볼 압력 시계열")

            metric_options = {
                "Signed 압력": "signed_pressure_score",
                "숏 후보 압력": "short_watch_pressure",
                "롱 후보 압력": "long_watch_pressure",
                "Plot score": "plot_score",
                "가격 변화율": "price_change_from_start_pct",
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

            pressure_history_df["_snapshot_time_plot"] = pd.to_datetime(
                pressure_history_df["_snapshot_time"],
                errors="coerce",
            ).dt.strftime("%Y-%m-%d %H:%M:%S")

            pressure_fig = px.line(
                pressure_history_df,
                x="_snapshot_time_plot",
                y=selected_pressure_metric,
                color="symbol",
                markers=True,
                hover_data=make_pressure_hover_data(pressure_history_df),
                title=(
                    f"{', '.join(timeseries_symbols)} | "
                    f"{selected_pressure_metric_label} | {pressure_range}"
                ),
                labels={
                    "_snapshot_time_plot": "Time KST",
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
                    x="_snapshot_time_plot",
                    y=selected_price_metric,
                    color="symbol",
                    markers=True,
                    hover_data=make_pressure_hover_data(pressure_history_df),
                    title=(
                        f"{', '.join(timeseries_symbols)} | "
                        f"{selected_price_metric_label} | {pressure_range}"
                    ),
                    labels={
                        "_snapshot_time_plot": "Time KST",
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
                "funding_rate",
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
        "funding_rate",
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
