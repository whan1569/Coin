# Coin 실행 방법

## 1. 프로젝트 다운로드

```powershell id="l8bm49"
git clone https://github.com/whan1569/Coin.git
cd Coin
```

## 2. 가상환경 생성

```powershell id="x9cmua"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. 패키지 설치

```powershell id="b4qv2k"
python -m pip install --upgrade pip
python -m pip install aiohttp numpy pandas plotly streamlit orjson tzdata
```

또는 `requirements.txt`를 사용할 경우:

```powershell id="hmsrhf"
python -m pip install -r requirements.txt
```

---

# 실행 방법

이 프로젝트는 **현재 스냅샷 분석**과 **시계열 히스토리 분석**으로 나뉩니다.

## 1. 현재 스냅샷 분석 실행

현재 시점의 LS / OI 데이터를 수집합니다.

```powershell id="wl5sir"
python LS_weights.py
```

실행 후 아래 CSV 파일들이 생성됩니다.

```text id="6boz6c"
binance_ls_lsoi_score.csv
spot_ok_lsoi.csv
spot_missing_lsoi.csv
```

그다음 현재 스냅샷 대시보드를 실행합니다.

```powershell id="pp6m5p"
python -m streamlit run view.py
```

---

## 2. 시계열 히스토리 분석 실행

15분 단위 히스토리 데이터를 수집합니다.

```powershell id="15eu45"
python LS_history_collector.py
```

기본 저장 위치는 아래와 같습니다.

```text id="0f2gtt"
data/history_lsoi_15m.csv
```

히스토리 대시보드를 실행합니다.

```powershell id="e1ta80"
python -m streamlit run view_history.py
```

---

# 전체 실행 순서

처음 실행할 때는 아래 순서로 하면 됩니다.

```powershell id="b08o7x"
git clone https://github.com/whan1569/Coin.git
cd Coin

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install aiohttp numpy pandas plotly streamlit orjson tzdata

python LS_weights.py
python LS_history_collector.py

python -m streamlit run view.py
python -m streamlit run view_history.py
```

---

# 파일별 역할

```text id="9m6j74"
LS_weights.py
```

현재 시점 기준 LS / OI 데이터를 수집하고 CSV를 생성합니다.

```text id="u9qajx"
view.py
```

`binance_ls_lsoi_score.csv`를 읽어서 현재 스냅샷 대시보드를 실행합니다.

```text id="f9jvfh"
LS_history_collector.py
```

15분 단위 히스토리 데이터를 수집하고 `data/history_lsoi_15m.csv`를 생성합니다.

```text id="6mxsvc"
view_history.py
```

`data/history_lsoi_15m.csv`를 읽어서 시계열 대시보드를 실행합니다.

---

# 참고

현재 분석만 볼 경우:

```powershell id="sdh7fk"
python LS_weights.py
python -m streamlit run view.py
```

시계열 분석까지 볼 경우:

```powershell id="hwp6af"
python LS_history_collector.py
python -m streamlit run view_history.py
```

둘 다 볼 경우:

```powershell id="1xn0k7"
python LS_weights.py
python LS_history_collector.py
python -m streamlit run view_history.py
```
