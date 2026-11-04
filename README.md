# Coin 실행 방법

## 1. 프로젝트 다운로드

```powershell
git clone https://github.com/whan1569/Coin.git
cd Coin
```

## 2. 가상환경 생성

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. 패키지 설치

```powershell
python -m pip install --upgrade pip
python -m pip install aiohttp numpy pandas plotly streamlit orjson tzdata
```

또는 `requirements.txt`를 사용할 경우:

```powershell
python -m pip install -r requirements.txt
```

---

# 실행 방법

이 프로젝트는 **현재 스냅샷 분석**, **시계열 히스토리 분석**, **누적 압력 매트릭스 분석**으로 나뉩니다.

## 1. 현재 스냅샷 분석 실행

현재 시점의 LS / OI 데이터를 수집합니다.

```powershell
python LS_weights.py
```

실행 후 아래 CSV 파일들이 생성됩니다.

```text
binance_ls_lsoi_score.csv
spot_ok_lsoi.csv
spot_missing_lsoi.csv
```

그다음 현재 스냅샷 대시보드를 실행합니다.

```powershell
python -m streamlit run view.py
```

---

## 2. 시계열 히스토리 분석 실행

15분 단위 히스토리 데이터를 수집합니다.

```powershell
python LS_history_collector.py
```

기본 저장 위치는 아래와 같습니다.

```text
data/history_lsoi_15m.csv
```

히스토리 대시보드를 실행합니다.

```powershell
python -m streamlit run view_history.py
```

---

## 3. 누적 압력 매트릭스 실행

`data/history_lsoi_15m.csv`를 이용해 1~120일 누적 압력 기준 2차원 매트릭스를 표시합니다.

```powershell
python -m streamlit run view_matrix.py
```

매트릭스 의미:

- X축: 선택기간 LS `heat_score` 누적 압력
- Y축: 누적 압력 대비 가격 변화율
- 우측상단: LONG 후보
- 좌측하단: SHORT 후보
- 버블 크기: Binance `sumOpenInterestValue` 기반 OI 명목가치의 상대 크기
- 색상: 선택기간 가격 범위 내 현재 위치(0%=저점, 100%=고점)

> 30일을 넘는 분석은 로컬 `history_lsoi_15m.csv`에 과거 데이터가 실제로 누적되어 있어야 합니다.

---

# 전체 실행 순서

처음 실행할 때는 아래 순서로 하면 됩니다.

```powershell
git clone https://github.com/whan1569/Coin.git
cd Coin

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install aiohttp numpy pandas plotly streamlit orjson tzdata

python LS_weights.py
python LS_history_collector.py

python -m streamlit run view.py
python -m streamlit run view_history.py
python -m streamlit run view_matrix.py
```

---

# 파일별 역할

`LS_weights.py`

현재 시점 기준 LS / OI 데이터를 수집하고 CSV를 생성합니다.

`view.py`

`binance_ls_lsoi_score.csv`를 읽어서 현재 스냅샷 대시보드를 실행합니다.

`LS_history_collector.py`

15분 단위 히스토리 데이터를 수집하고 `data/history_lsoi_15m.csv`를 생성합니다.

`view_history.py`

`data/history_lsoi_15m.csv`를 읽어서 시계열 대시보드를 실행합니다.

`view_matrix.py`

선택기간 LS 누적 압력, 가격 반응, OI 명목가치, 기간 내 가격 위치를 2차원 버블 차트로 표시합니다.

---

---

# GMT Coin — 코인별 현물·선물 체결 압력 (1차)

기존 LS/OI/펀딩 과열 점수는 유지하고, **동일한 15분 구간**의 현물과
USDT-M 무기한 선물에서 **매수 주도/매도 주도 거래대금 불균형**을 별도로 계산합니다.
이는 거래소의 체결 주도 방향이지, 순자금 유입이나 롱/숏 미결제 포지션 변화가 아닙니다.

### 실행

기존 설치 후 아래 명령을 실행하세요.

```powershell
python LS_history_collector.py --days 20 --retain-days 120
python -m streamlit run view_pressure.py
```

- `--days`는 Binance LS/OI API에서 새로 조회할 과거 구간(1~29일, 기본 20일)입니다.
- `--retain-days`는 기존 파일에서 **보존할 데이터 기간**(기본 120일)입니다.
  120일치를 한 번에 역조회한다는 의미가 아닙니다. 수집기를 정기 실행해
  데이터를 누적해야 120일 범위를 관측할 수 있습니다.
- 기존 CSV는 그대로 읽으며, 과거 데이터에 현물 매수 주도 거래대금이
  없을 때는 `USDTM_ONLY`로 표시됩니다. 없는 값을 0으로 대체하지 않습니다.
- 압력 계산에 사용할 현물/선물 15분봉은 동일한 **시작 시각**과
  같은 **USDT 거래대금 단위**를 사용합니다. 다른 시각의 현물 봉을 가져와
  현재 선물 값과 합치지 않습니다.
- `--no-spot-klines` 옵션은 현물 압력 수집을 생략합니다.
- 일부 선물 시장정보 API가 API 키 헤더를 요구하는 경우, 키를 소스에 넣지 말고
  PowerShell에서 `$env:BINANCE_API_KEY="발급받은_API_KEY"`로 지정하세요.
  실제 권한과 API 이용 정책은 Binance 계정/지역별로 확인해야 합니다.

### 주요 신규 컬럼

| 컬럼 | 의미 |
|---|---|
| `spot_trade_pressure_pct` | 현물 (매수 주도 - 매도 주도) / 현물 거래대금 × 100 |
| `futures_trade_pressure_pct` | USDT-M 무기한 선물에 같은 식을 적용 |
| `spot_trade_volume_share_pct` | 실제 유효 거래대금 기준 현물 비중 |
| `futures_trade_volume_share_pct` | 실제 유효 거래대금 기준 선물 비중 |
| `trade_pressure_turnover_weighted_pct` | 현물과 선물 순매수주도 거래대금 합 / 전체 거래대금 × 100 |
| `trade_pressure_balanced_pct` | 현물·선물 각각 50% 가중 (두 시장 모두 있을 때만) |
| `trade_pressure_coverage` | 두 시장 관측 / 현물만 / 선물만 / 결측 구분 |
| `trade_pressure_scope` | 현재 포함된 시장 범위를 명확히 표시 |

### 설계 범위와 다음 확장

현재 통합 압력은 **Binance 현물 USDT 거래쌍 + USDT-M 무기한 선물**
두 시장만 포함합니다. 기존 L/S·OI 분석도 계속 USDT-M 무기한 선물 기준입니다.

아직 통합 계산에 포함하지 않은 시장:
USDC-M, COIN-M, 분기·인도 선물, 옵션, 타 거래소.

COIN-M API의 봉 거래량에는 **계약 수량과 기초자산 수량**이 섞여 있으므로,
USDT-M의 USDT 표시 거래대금과 그대로 더하면 안 됩니다.
추가할 때 계약 단위·통화 단위·가격 시점·동일 자산 매핑을 검증하고
시장별 유효 데이터 및 커버리지 표시를 확장해야 합니다.
선물 미결제약정(OI)과 롱숏 비율도 체결대금 가중치에 직접 섞지 않고 별도 지표로 유지합니다.

검증은 네트워크 없이 수행할 수 있습니다.

```powershell
python -m unittest discover -s tests -v
```

상용 재배포 전에는 데이터 사용 권한, 지역별 API 접근, 누락·지연·시각 정합성,
실데이터 검증을 별도로 확인해야 합니다.


# 참고

현재 분석만 볼 경우:

```powershell
python LS_weights.py
python -m streamlit run view.py
```

시계열 분석까지 볼 경우:

```powershell
python LS_history_collector.py
python -m streamlit run view_history.py
```

누적 압력 매트릭스를 볼 경우:

```powershell
python LS_history_collector.py
python -m streamlit run view_matrix.py
```
