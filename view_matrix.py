import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from view_history import HISTORY_CSV_PATH, load_history_raw


SAMPLES_PER_DAY_15M = 96
MAX_ANALYSIS_DAYS = 120


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _first_valid(series: pd.Series):
    valid = series.dropna()
    return np.nan if valid.empty else valid.iloc[0]


def _last_valid(series: pd.Series):
    valid = series.dropna()
    return np.nan if valid.empty else valid.iloc[-1]


def _pct_change(start, end) -> float:
    if pd.isna(start) or pd.isna(end) or start == 0:
        return np.nan
    return (end / start - 1.0) * 100.0


def _price_position(low, high, current) -> float:
    if pd.isna(low) or pd.isna(high) or pd.isna(current):
        return np.nan
    if high <= low:
        return 50.0
    return float(np.clip((current - low) / (high - low) * 100.0, 0.0, 100.0))


def _quadrant(pressure: float, response: float) -> str:
    if pd.isna(pressure) or pd.isna(response):
        return "NEUTRAL"
    if pressure > 0 and response > 0:
        return "LONG_CANDIDATE"
    if pressure < 0 and response < 0:
        return "SHORT_CANDIDATE"
    if pressure < 0 and response > 0:
        return "SHORT_SQUEEZE"
    if pressure > 0 and response < 0:
        return "LONG_TRAP"
    return "NEUTRAL"


def build_pressure_matrix(
    history_raw: pd.DataFrame,
    end_time,
    analysis_days: int,
    min_coverage: float,
) -> tuple[pd.DataFrame, dict]:
    df = history_raw.copy()

    if df.empty:
        return pd.DataFrame(), {}

    end_ts = pd.Timestamp(end_time).floor("min")
    start_ts = end_ts - pd.Timedelta(days=int(analysis_days))

    df = df[
        (df["_snapshot_time"] >= start_ts)
        & (df["_snapshot_time"] <= end_ts)
    ].copy()

    if df.empty:
        return pd.DataFrame(), {
            "start_time": start_ts,
            "end_time": end_ts,
            "available_days": 0.0,
        }

    required = ["symbol", "heat_score", "mark_price"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"히스토리 CSV에 필요한 컬럼이 없습니다: {missing}")

    numeric_cols = [
        "heat_score",
        "mark_price",
        "open_interest",
        "oi_nv",
        "ls_ratio",
        "ls_acco",
        "ls_position",
        "funding_rate_8h_pct",
        "funding_rate",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = _numeric(df[col])

    if "oi_nv" not in df.columns:
        df["oi_nv"] = np.nan

    if "open_interest" in df.columns:
        fallback_oi_nv = df["open_interest"] * df["mark_price"]
        df["oi_nv"] = df["oi_nv"].fillna(fallback_oi_nv)

    expected_samples = max(1, int(analysis_days) * SAMPLES_PER_DAY_15M)
    rows = []

    for symbol, sub in df.groupby("symbol", sort=False):
        sub = sub.sort_values("_snapshot_time").copy()
        heat = _numeric(sub["heat_score"]).dropna()
        price = _numeric(sub["mark_price"]).dropna()
        oi_nv = _numeric(sub["oi_nv"]).dropna()

        if len(heat) < 2 or len(price) < 2:
            continue

        sample_count = int(heat.shape[0])
        coverage = min(sample_count / expected_samples, 1.0)

        if coverage < float(min_coverage):
            continue

        start_price = float(price.iloc[0])
        end_price = float(price.iloc[-1])
        low_price = float(price.min())
        high_price = float(price.max())
        price_change_pct = _pct_change(start_price, end_price)
        price_position_pct = _price_position(low_price, high_price, end_price)

        # 15분 heat_score를 시간 적분한 값.
        # sum / 96 = "heat-score day"이므로 같은 기간 안에서 누적 방향/강도를 비교하기 쉽다.
        pressure_cum_raw = float(heat.sum() / SAMPLES_PER_DAY_15M)
        pressure_mean = float(heat.mean())

        oi_start = float(oi_nv.iloc[0]) if not oi_nv.empty else np.nan
        oi_end = float(oi_nv.iloc[-1]) if not oi_nv.empty else np.nan
        oi_change_pct = _pct_change(oi_start, oi_end)

        row = {
            "symbol": str(symbol).upper().strip(),
            "sample_count": sample_count,
            "coverage_pct": coverage * 100.0,
            "pressure_cum_raw": pressure_cum_raw,
            "pressure_mean": pressure_mean,
            "price_start": start_price,
            "price_now": end_price,
            "price_low": low_price,
            "price_high": high_price,
            "price_change_pct": price_change_pct,
            "price_position_pct": price_position_pct,
            "oi_notional_usdt": oi_end,
            "oi_change_pct": oi_change_pct,
        }

        for col in ["ls_ratio", "ls_acco", "ls_position", "funding_rate_8h_pct", "funding_rate"]:
            if col in sub.columns:
                values = _numeric(sub[col]).dropna()
                row[f"{col}_mean"] = float(values.mean()) if not values.empty else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out, {
            "start_time": start_ts,
            "end_time": end_ts,
            "available_days": 0.0,
        }

    pressure_abs = out["pressure_cum_raw"].abs().replace(0, np.nan).dropna()
    pressure_floor = float(pressure_abs.quantile(0.25)) if not pressure_abs.empty else 1.0
    pressure_floor = max(pressure_floor, 1.0)

    out["pressure_efficiency_raw"] = (
        out["price_change_pct"]
        / out["pressure_cum_raw"].abs().clip(lower=pressure_floor)
    )

    response_abs = out["pressure_efficiency_raw"].abs().replace(0, np.nan).dropna()
    response_cap = float(response_abs.quantile(0.98)) if not response_abs.empty else np.nan

    if pd.isna(response_cap) or response_cap <= 0:
        out["pressure_efficiency"] = out["pressure_efficiency_raw"]
    else:
        out["pressure_efficiency"] = out["pressure_efficiency_raw"].clip(
            lower=-response_cap,
            upper=response_cap,
        )

    valid_oi = out["oi_notional_usdt"].where(out["oi_notional_usdt"] > 0)
    if valid_oi.notna().any():
        # OI 원수량이 아니라 Binance sumOpenInterestValue(USDT 명목가치)를 사용한다.
        # 버블 면적은 절대 규모의 단조 변환인 percentile로 눌러 BTC 등 초대형 종목 독점을 막는다.
        oi_rank = valid_oi.rank(method="average", pct=True)
        out["oi_size"] = (5.0 + 95.0 * oi_rank).fillna(5.0)
    else:
        out["oi_size"] = 10.0

    out["quadrant"] = out.apply(
        lambda row: _quadrant(row["pressure_cum_raw"], row["pressure_efficiency"]),
        axis=1,
    )

    out["candidate_strength"] = (
        np.sqrt(
            out["pressure_cum_raw"].abs()
            * out["price_change_pct"].abs().fillna(0)
        )
        * (out["coverage_pct"] / 100.0)
    )

    actual_min = pd.Timestamp(df["_snapshot_time"].min())
    available_days = max(0.0, (end_ts - actual_min).total_seconds() / 86400.0)

    meta = {
        "start_time": start_ts,
        "end_time": end_ts,
        "available_days": available_days,
        "pressure_floor": pressure_floor,
        "response_cap": response_cap,
    }
    return out, meta


def main():
    st.set_page_config(page_title="Crypto Pressure Matrix", layout="wide")
    st.title("Crypto Pressure Accumulation Matrix")
    st.caption(
        "우측상단=LONG 후보 / 좌측하단=SHORT 후보. "
        "X는 LS heat_score 누적, Y는 누적 압력 대비 가격 변화율, "
        "버블 크기는 가격 보정된 OI 명목가치(oi_nv), 색은 선택기간 내 현재 가격 위치입니다."
    )

    with st.sidebar:
        st.header("Matrix 설정")
        history_csv_path = st.text_input("히스토리 CSV", value=HISTORY_CSV_PATH)

        analysis_days = st.slider(
            "누적 분석 기간(일)",
            min_value=1,
            max_value=MAX_ANALYSIS_DAYS,
            value=20,
            step=1,
        )

        min_coverage_pct = st.slider(
            "최소 데이터 커버리지(%)",
            min_value=20,
            max_value=100,
            value=60,
            step=5,
        )

        candidate_filter = st.selectbox(
            "영역 필터",
            [
                "ALL",
                "LONG_CANDIDATE",
                "SHORT_CANDIDATE",
                "SHORT_SQUEEZE",
                "LONG_TRAP",
            ],
            index=0,
        )

        sort_mode = st.selectbox(
            "정렬 기준",
            [
                "후보 강도",
                "OI 명목가치",
                "누적 압력 절댓값",
                "가격 변화율 절댓값",
            ],
            index=0,
        )

        display_count = st.number_input(
            "표시 개수",
            min_value=10,
            max_value=1000,
            value=150,
            step=10,
        )

        search_text = st.text_input(
            "심볼 검색/라벨",
            value="",
            placeholder="BTC ETH SOL",
        )

        show_all_labels = st.checkbox("전체 심볼명 표시", value=False)

    if not os.path.exists(history_csv_path):
        st.error(f"히스토리 CSV를 찾지 못했습니다: {history_csv_path}")
        st.code("python LS_history_collector.py\npython -m streamlit run view_matrix.py", language="powershell")
        return

    try:
        history_raw = load_history_raw(history_csv_path)
    except Exception as exc:
        st.error(f"히스토리 CSV 로드 실패: {exc}")
        return

    all_times = sorted(pd.to_datetime(history_raw["_snapshot_time"].dropna().unique()).tolist())
    if not all_times:
        st.warning("사용 가능한 스냅샷 시간이 없습니다.")
        return

    latest_time = pd.Timestamp(all_times[-1])
    earliest_time = pd.Timestamp(all_times[0])

    with st.sidebar:
        end_time = st.select_slider(
            "분석 종료 시점",
            options=all_times,
            value=all_times[-1],
            format_func=lambda x: pd.Timestamp(x).strftime("%m-%d %H:%M"),
        )
        st.caption(
            f"로컬 보유 히스토리: {earliest_time.strftime('%Y-%m-%d %H:%M')} ~ "
            f"{latest_time.strftime('%Y-%m-%d %H:%M')} KST"
        )

    matrix_df, meta = build_pressure_matrix(
        history_raw=history_raw,
        end_time=end_time,
        analysis_days=int(analysis_days),
        min_coverage=float(min_coverage_pct) / 100.0,
    )

    if matrix_df.empty:
        st.warning("해당 기간/커버리지 조건으로 계산 가능한 심볼이 없습니다.")
        return

    available_days = float(meta.get("available_days", 0.0))
    if available_days + 0.25 < analysis_days:
        st.warning(
            f"요청 기간은 {analysis_days}일이지만 종료시점 기준 로컬 CSV에 약 {available_days:.1f}일만 있습니다. "
            "120일 분석은 데이터를 로컬에 계속 누적한 뒤 사용할 수 있습니다."
        )

    # Binance openInterestHist는 API 자체가 최근 1개월만 역조회할 수 있으므로,
    # 30일을 넘는 구간은 로컬 CSV에 과거부터 보존된 데이터가 있어야 한다.
    if analysis_days > 30:
        st.info(
            "30일 초과 구간은 Binance OI API에서 과거를 한 번에 백필할 수 없습니다. "
            "로컬 history_lsoi_15m.csv에 누적 보존된 구간만 실제 계산에 사용됩니다."
        )

    if candidate_filter != "ALL":
        matrix_df = matrix_df[matrix_df["quadrant"] == candidate_filter].copy()

    if matrix_df.empty:
        st.warning("선택한 영역에 표시할 심볼이 없습니다.")
        return

    if sort_mode == "후보 강도":
        matrix_df = matrix_df.sort_values("candidate_strength", ascending=False)
    elif sort_mode == "OI 명목가치":
        matrix_df = matrix_df.sort_values("oi_notional_usdt", ascending=False, na_position="last")
    elif sort_mode == "누적 압력 절댓값":
        matrix_df = matrix_df.assign(_sort=matrix_df["pressure_cum_raw"].abs()).sort_values("_sort", ascending=False)
    else:
        matrix_df = matrix_df.assign(_sort=matrix_df["price_change_pct"].abs()).sort_values("_sort", ascending=False)

    matrix_df = matrix_df.head(int(display_count)).copy()

    search_tokens = [token.upper().strip() for token in search_text.replace(",", " ").split() if token.strip()]
    search_symbols = set()
    for token in search_tokens:
        if token.endswith("USDT"):
            search_symbols.add(token)
        else:
            search_symbols.add(token + "USDT")

    matrix_df["label_symbol"] = ""
    if show_all_labels:
        matrix_df["label_symbol"] = matrix_df["symbol"]
    elif search_symbols:
        matrix_df.loc[matrix_df["symbol"].isin(search_symbols), "label_symbol"] = matrix_df["symbol"]

    q_counts = matrix_df["quadrant"].value_counts()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("표시", len(matrix_df))
    m2.metric("LONG 후보", int(q_counts.get("LONG_CANDIDATE", 0)))
    m3.metric("SHORT 후보", int(q_counts.get("SHORT_CANDIDATE", 0)))
    m4.metric("SHORT squeeze", int(q_counts.get("SHORT_SQUEEZE", 0)))
    m5.metric("LONG trap", int(q_counts.get("LONG_TRAP", 0)))

    hover_data = {
        "symbol": True,
        "quadrant": True,
        "pressure_cum_raw": ":.3f",
        "pressure_mean": ":.3f",
        "pressure_efficiency_raw": ":.4f",
        "price_change_pct": ":.2f",
        "price_position_pct": ":.1f",
        "price_now": ":.8f",
        "price_low": ":.8f",
        "price_high": ":.8f",
        "oi_notional_usdt": ":,.0f",
        "oi_change_pct": ":.2f",
        "coverage_pct": ":.1f",
        "sample_count": True,
        "oi_size": False,
        "label_symbol": False,
    }

    for optional_col, fmt in [
        ("ls_ratio_mean", ":.3f"),
        ("ls_acco_mean", ":.3f"),
        ("ls_position_mean", ":.3f"),
        ("funding_rate_8h_pct_mean", ":.4f"),
        ("funding_rate_mean", ":.6f"),
    ]:
        if optional_col in matrix_df.columns:
            hover_data[optional_col] = fmt

    fig = px.scatter(
        matrix_df,
        x="pressure_cum_raw",
        y="pressure_efficiency",
        size="oi_size",
        size_max=58,
        color="price_position_pct",
        color_continuous_scale="RdBu_r",
        range_color=[0, 100],
        text="label_symbol",
        custom_data=["symbol"],
        hover_data=hover_data,
        labels={
            "pressure_cum_raw": "누적 LS 압력 (heat-score day)",
            "pressure_efficiency": "가격 변화율 / |누적 압력|",
            "price_position_pct": "기간 내 가격 위치 (%)",
            "oi_size": "OI 규모",
        },
        title=(
            f"{analysis_days}일 Pressure Matrix | "
            f"{pd.Timestamp(end_time).strftime('%Y-%m-%d %H:%M')} KST"
        ),
    )

    fig.add_vline(x=0, line_dash="dash", line_width=1)
    fig.add_hline(y=0, line_dash="dash", line_width=1)
    fig.update_traces(textposition="top center", marker=dict(opacity=0.75))

    x_abs = pd.to_numeric(matrix_df["pressure_cum_raw"], errors="coerce").abs().max()
    y_abs = pd.to_numeric(matrix_df["pressure_efficiency"], errors="coerce").abs().max()
    x_abs = 1.0 if pd.isna(x_abs) or x_abs <= 0 else float(x_abs) * 1.12
    y_abs = 1.0 if pd.isna(y_abs) or y_abs <= 0 else float(y_abs) * 1.18

    fig.update_xaxes(range=[-x_abs, x_abs], zeroline=False)
    fig.update_yaxes(range=[-y_abs, y_abs], zeroline=False)

    fig.add_annotation(x=x_abs * 0.72, y=y_abs * 0.88, text="LONG 후보", showarrow=False)
    fig.add_annotation(x=-x_abs * 0.72, y=-y_abs * 0.88, text="SHORT 후보", showarrow=False)
    fig.add_annotation(x=-x_abs * 0.72, y=y_abs * 0.88, text="SHORT SQUEEZE", showarrow=False)
    fig.add_annotation(x=x_abs * 0.72, y=-y_abs * 0.88, text="LONG TRAP", showarrow=False)

    fig.update_layout(height=780, coloraxis_colorbar=dict(title="가격 위치 %"))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "버블 크기: oi_nv(sumOpenInterestValue, USDT 명목가치)의 표시 종목 내 percentile. "
        "색: 0%=선택기간 저점, 100%=선택기간 고점. "
        "Y는 압력이 매우 작은 종목의 폭주를 막기 위해 압력 하위 25% 값을 분모 바닥으로 사용하고 98% 꼬리를 시각화용으로 제한합니다."
    )

    table_cols = [
        "symbol",
        "quadrant",
        "pressure_cum_raw",
        "pressure_mean",
        "price_change_pct",
        "pressure_efficiency_raw",
        "price_position_pct",
        "oi_notional_usdt",
        "oi_change_pct",
        "coverage_pct",
        "sample_count",
        "candidate_strength",
    ]
    table_cols += [
        col for col in [
            "ls_ratio_mean",
            "ls_acco_mean",
            "ls_position_mean",
            "funding_rate_8h_pct_mean",
        ]
        if col in matrix_df.columns
    ]

    st.subheader("Matrix 데이터")
    st.dataframe(matrix_df[table_cols], use_container_width=True, height=520)


if __name__ == "__main__":
    main()
