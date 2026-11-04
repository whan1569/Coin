"""Coin per-symbol executed trade pressure dashboard (Binance spot + USDT-M)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from market_pressure import add_trade_pressure_features

DATA_PATH = "data/history_lsoi_15m.csv"


@st.cache_data(show_spinner=False)
def load_pressure_history(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if "timestamp_ms" not in df.columns or "symbol" not in df.columns:
        raise ValueError("히스토리 파일에 timestamp_ms 또는 symbol이 없습니다.")
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df = df.dropna(subset=["timestamp_ms", "symbol"]).copy()
    if "spot_taker_buy_quote_volume_15m" not in df.columns:
        st.warning("기존 CSV에는 현물 매수 주도 거래대금이 없습니다. 새 수집 이후 두 시장 압력이 표시됩니다.")
    df = add_trade_pressure_features(df)
    df["snapshot_kst"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_convert("Asia/Seoul")
    return df


def main() -> None:
    st.set_page_config(page_title="GMT Coin · Trade Pressure", layout="wide")
    st.title("GMT Coin · 체결 압력")
    st.caption("Binance 현물 + USDT-M 무기한 (USDT 거래쌍), 15분봉 기준. 롱·숏 포지션 비율이나 OI와 별도 지표입니다.")

    path = st.sidebar.text_input("시계열 데이터 경로", value=DATA_PATH)
    if not os.path.isfile(path):
        st.info("먼저 python LS_history_collector.py 로 히스토리를 수집하세요.")
        return
    try:
        history = load_pressure_history(path)
    except Exception as exc:
        st.error(f"데이터 로드 실패: {exc}")
        return
    if history.empty:
        st.warning("유효한 히스토리 행이 없습니다.")
        return

    timestamps = sorted(history["timestamp_ms"].unique())
    selected_ts = st.sidebar.select_slider(
        "스냅샷 시간 (KST)",
        options=timestamps,
        value=timestamps[-1],
        format_func=lambda ms: pd.to_datetime(ms, unit="ms", utc=True).tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M"),
    )
    coverage_filter = st.sidebar.selectbox(
        "집계 범위", ["전체", "SPOT_AND_USDTM", "SPOT_ONLY", "USDTM_ONLY"]
    )
    max_rows = st.sidebar.slider("표시 종목 수", 10, 200, 50, step=10)
    snapshot = history[history["timestamp_ms"] == selected_ts].copy()
    if coverage_filter != "전체":
        snapshot = snapshot[snapshot["trade_pressure_coverage"] == coverage_filter]
    snapshot = snapshot[snapshot["trade_pressure_turnover_weighted_pct"].notna()]
    if snapshot.empty:
        st.warning("이 시간에 사용할 수 있는 체결 압력 데이터가 없습니다.")
        return
    snapshot["abs_pressure"] = snapshot["trade_pressure_turnover_weighted_pct"].abs()
    snapshot = snapshot.sort_values("abs_pressure", ascending=False).head(max_rows)
    full_count = int((snapshot["trade_pressure_coverage"] == "SPOT_AND_USDTM").sum())
    st.caption(f"표시 종목 {len(snapshot)}개 · 현물/선물 모두 유효한 종목 {full_count}개 · 나머지는 부분 집계")

    fig = px.scatter(
        snapshot,
        x="spot_trade_pressure_pct",
        y="futures_trade_pressure_pct",
        size="trade_pressure_total_quote_volume_15m",
        color="trade_pressure_turnover_weighted_pct",
        hover_name="symbol",
        hover_data=["trade_pressure_coverage", "spot_trade_volume_share_pct", "futures_trade_volume_share_pct"],
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title="현물 vs 선물 매수 주도 체결 압력 · 완전 집계 종목만 위치 표시",
        labels={"spot_trade_pressure_pct": "현물 압력 (%)", "futures_trade_pressure_pct": "선물 압력 (%)"},
    )
    fig.add_hline(y=0, line_dash="dot")
    fig.add_vline(x=0, line_dash="dot")
    st.plotly_chart(fig, use_container_width=True)

    columns = [
        "symbol", "trade_pressure_coverage", "spot_trade_pressure_pct",
        "futures_trade_pressure_pct", "spot_trade_volume_share_pct",
        "futures_trade_volume_share_pct", "trade_pressure_turnover_weighted_pct",
        "trade_pressure_balanced_pct", "trade_pressure_total_quote_volume_15m",
        "heat_score", "oi_nv",
    ]
    st.dataframe(snapshot[[c for c in columns if c in snapshot.columns]], hide_index=True, use_container_width=True)

    symbol = st.selectbox("코인별 시계열", sorted(snapshot["symbol"].unique()))
    series = history[history["symbol"] == symbol].sort_values("timestamp_ms").tail(96 * 7)
    series_long = series.melt(
        id_vars="snapshot_kst",
        value_vars=["spot_trade_pressure_pct", "futures_trade_pressure_pct", "trade_pressure_turnover_weighted_pct", "trade_pressure_balanced_pct"],
        var_name="압력 유형", value_name="압력 (%)",
    )
    line = px.line(series_long, x="snapshot_kst", y="압력 (%)", color="압력 유형", title=f"{symbol} · 최대 최근 7일")
    st.plotly_chart(line, use_container_width=True)
    st.caption("거래대금 가중 = (현물 순매수주도 USDT + 선물 순매수주도 USDT) / 유효한 두 시장 거래대금. 균형 = 두 시장 압력의 산술평균(둘 다 유효할 때만).")
    st.caption("해당 거래소의 해당 거래쌍에 한정된 체결 불균형이며 순자금 유입, 투자 방향 또는 전체 거래소 압력을 의미하지 않습니다.")


if __name__ == "__main__":
    main()
