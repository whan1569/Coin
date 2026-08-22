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
