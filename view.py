# view.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# 설정
# =========================================================

CSV_PATH = "binance_ls_lsoi_score.csv"


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
# 2. 검색 보조
# =========================================================

def resolve_search_symbols(
    search_text: str,
    available_symbols: list[str],
) -> tuple[list[str], list[str]]:
    """
    CSV에 실제 존재하는 심볼 기준으로 검색어를 해석한다.

    예:
    - XMR      -> XMRUSDT
    - XMRUSDT  -> XMRUSDT
    - vv       -> VV로 시작하는 USDT 심볼
    - XMR, LIT -> XMRUSDT, LITUSDT
    """
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
    """
    Streamlit plotly selection 이벤트에서 선택된 symbol 추출.
    Streamlit 버전에 따라 chart_event 구조가 다를 수 있어서 방어적으로 처리.
    """
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
# 3. 기준 CSV 로드
# =========================================================

@st.cache_data
def load_lsoi_score(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

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
        raise ValueError(f"{csv_path}에 필요한 컬럼이 없음: {missing}")

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

    # -----------------------------------------------------
    # direction / agreement / overheat_abs
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # spot_market_category
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # watch_side
    # -----------------------------------------------------
    if "watch_side" not in df.columns:
        df["watch_side"] = df.apply(make_watch_side, axis=1)

    # -----------------------------------------------------
    # 구버전 CSV 호환용 최소 계산
    # 새 스캐너 CSV면 값 그대로 사용
    # -----------------------------------------------------
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


# =========================================================
# 4. 필터 / 정렬
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
# 5. Streamlit 화면
# =========================================================

def main():
    st.set_page_config(
        page_title="Binance LS × OI Viewer",
        layout="wide",
    )

    st.title("Binance LS × OI Pressure Viewer")

    st.caption(
        "기준 데이터는 binance_ls_lsoi_score.csv를 사용합니다. "
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

        if st.button("새로고침"):
            st.cache_data.clear()
            st.rerun()

    # -----------------------------------------------------
    # 데이터 로드
    # -----------------------------------------------------
    try:
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

    # -----------------------------------------------------
    # 검색 심볼 별도 강조 표시
    # -----------------------------------------------------
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

    fig.update_layout(
        height=760,
    )

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

    active_symbols = list(
        dict.fromkeys(search_symbols + clicked_symbols)
    )

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

            detail_cols = [
                col for col in detail_cols
                if col in active_detail_df.columns
            ]

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

    st.download_button(
        label="현재 결과 CSV 다운로드",
        data=view[available_cols].to_csv(index=False, encoding="utf-8-sig"),
        file_name=(
            f"{market_category}_{direction_filter}_"
            f"{rank_mode}_{int(display_count)}.csv"
        ),
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
