<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
body { font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif; }
code { font-family: Consolas, monospace; }
</style>

# 🐣 파이썬 금융논문 완전 초보 가이드

> AI 없이 혼자서 금융 논문과 분석 코드를 작성하기 위한 완전 초보자용 가이드
> 
> 이 가이드는 "초딩한테 알려주듯" 하나하나 설명합니다. 천천히 따라오세요!

---

## 📚 목차

1. [파이썬 기초 - 진짜 기초부터](#1-파이썬-기초---진짜-기초부터)
2. [pandas - 데이터 다루기](#2-pandas---데이터-다루기)
3. [금융 데이터 분석 기초](#3-금융-데이터-분석-기초)
4. [아이디어 → 논문 흐름](#4-아이디어--논문-흐름)
5. [BACBB 코드 한줄한줄 해설](#5-bacbb-코드-한줄한줄-해설)
6. [통계 검정 방법](#6-통계-검정-방법)
7. [그래프와 표 만들기](#7-그래프와-표-만들기)
8. [실전 팁과 자주 하는 실수](#8-실전-팁과-자주-하는-실수)

---

## 1. 파이썬 기초 - 진짜 기초부터

### 1.1 변수란?

변수는 **값을 담는 상자**예요. 상자에 이름표를 붙여서 나중에 꺼내 쓸 수 있어요.

```python
# 숫자 담기
price = 100          # price라는 상자에 100을 넣음
return_rate = 0.05   # return_rate라는 상자에 0.05를 넣음

# 문자 담기
coin_name = "BTC"    # coin_name이라는 상자에 "BTC"를 넣음

# 계산하기
new_price = price * (1 + return_rate)  # 100 * 1.05 = 105
print(new_price)     # 화면에 105 출력
```

**💡 핵심 포인트:**
- `=`는 "같다"가 아니라 "넣는다"라는 뜻!
- 변수 이름은 영어로, 의미 있게 짓기 (예: `price`, `return_rate`)
- 띄어쓰기 대신 `_` 사용 (예: `coin_name`)

### 1.2 리스트 (List)

리스트는 **여러 값을 순서대로 담는 상자**예요.

```python
# 코인 이름들을 리스트로
coins = ["BTC", "ETH", "XRP", "SOL"]

# 첫 번째 코인 꺼내기 (0부터 시작!)
first_coin = coins[0]   # "BTC"
print(first_coin)

# 두 번째 코인
second_coin = coins[1]  # "ETH"

# 마지막 코인 (-1은 뒤에서 첫 번째)
last_coin = coins[-1]   # "SOL"

# 리스트 길이 (몇 개 있나?)
count = len(coins)      # 4
```

**💡 핵심 포인트:**
- 인덱스는 0부터 시작! (첫 번째 = 0, 두 번째 = 1)
- `len()`으로 개수 세기
- `-1`은 마지막, `-2`는 뒤에서 두 번째

### 1.3 딕셔너리 (Dictionary)

딕셔너리는 **이름표가 붙은 서랍장**이에요. 이름으로 값을 찾을 수 있어요.

```python
# 코인별 가격
prices = {
    "BTC": 50000,
    "ETH": 3000,
    "XRP": 0.5
}

# BTC 가격 꺼내기
btc_price = prices["BTC"]  # 50000

# 새 코인 추가
prices["SOL"] = 100

# 모든 코인 이름 보기
coin_names = list(prices.keys())  # ["BTC", "ETH", "XRP", "SOL"]
```

### 1.4 반복문 (for loop)

같은 작업을 여러 번 반복할 때 사용해요.

```python
# 모든 코인 이름 출력하기
coins = ["BTC", "ETH", "XRP"]

for coin in coins:
    print(coin)
# 출력:
# BTC
# ETH
# XRP

# 각 코인의 수익률 계산하기
returns = [0.05, -0.02, 0.10]

for i, ret in enumerate(returns):
    print(f"{i}번째 수익률: {ret * 100}%")
# 출력:
# 0번째 수익률: 5.0%
# 1번째 수익률: -2.0%
# 2번째 수익률: 10.0%
```

**💡 핵심 포인트:**
- `for 변수 in 리스트:` 형태로 사용
- 들여쓰기(스페이스 4칸)가 중요! 들여쓴 부분이 반복됨
- `enumerate()`를 쓰면 순서 번호도 같이 받을 수 있음

### 1.5 조건문 (if)

조건에 따라 다른 행동을 할 때 사용해요.

```python
return_rate = 0.05

if return_rate > 0:
    print("수익!")
elif return_rate < 0:
    print("손실...")
else:
    print("본전")

# 한 줄로 쓰기 (삼항 연산자)
result = "수익" if return_rate > 0 else "손실"
```

### 1.6 함수 (Function)

자주 쓰는 코드를 묶어서 이름 붙여놓은 거예요.

```python
# 함수 만들기
def calculate_return(buy_price, sell_price):
    """
    수익률을 계산하는 함수
    
    Parameters:
    - buy_price: 매수 가격
    - sell_price: 매도 가격
    
    Returns:
    - 수익률 (소수점)
    """
    return_rate = (sell_price - buy_price) / buy_price
    return return_rate

# 함수 사용하기
my_return = calculate_return(100, 110)  # 0.1 (10%)
print(f"수익률: {my_return * 100}%")
```

**💡 핵심 포인트:**
- `def 함수이름(입력값):` 형태로 만듦
- `return`으로 결과 돌려줌
- `"""` 안에 설명 쓰면 나중에 뭐하는 함수인지 알 수 있음



---

## 2. pandas - 데이터 다루기

pandas는 **엑셀 같은 표 데이터를 다루는 도구**예요. 금융 분석의 핵심!

### 2.1 pandas 불러오기

```python
import pandas as pd   # pandas를 pd라는 별명으로 부름
import numpy as np    # numpy를 np라는 별명으로 부름 (수학 계산용)
```

**💡 왜 별명을 쓰나요?**
- `pandas.read_csv()` 대신 `pd.read_csv()`로 짧게 쓸 수 있어요
- 전 세계 개발자들이 다 이렇게 써서 약속처럼 됐어요

### 2.2 데이터 불러오기

```python
# CSV 파일 불러오기
prices = pd.read_csv('04_daily_prices.csv')

# 처음 5줄 보기
print(prices.head())

# 마지막 5줄 보기
print(prices.tail())

# 데이터 정보 보기
print(prices.info())

# 데이터 크기 (행, 열)
print(prices.shape)  # (1000, 50) = 1000행, 50열
```

### 2.3 날짜를 인덱스로 설정하기

금융 데이터는 날짜가 핵심이에요!

```python
# 날짜 컬럼을 인덱스로 설정
prices = pd.read_csv('04_daily_prices.csv', 
                     index_col=0,        # 첫 번째 열을 인덱스로
                     parse_dates=True)   # 날짜로 인식하게

# 이제 날짜로 데이터 찾기 가능
print(prices.loc['2024-01-01'])  # 2024년 1월 1일 데이터
```

**💡 index_col=0 이 뭐예요?**
- CSV 파일의 첫 번째 열(0번)을 행 이름(인덱스)으로 쓰겠다는 뜻
- 보통 첫 번째 열에 날짜가 있어요

### 2.4 특정 열/행 선택하기

```python
# 특정 열 선택 (BTC 가격만)
btc_prices = prices['BTC']

# 여러 열 선택
selected = prices[['BTC', 'ETH', 'XRP']]

# 특정 날짜 범위 선택
prices_2024 = prices['2024-01-01':'2024-12-31']

# 조건으로 선택 (BTC가 50000 이상인 날만)
high_btc = prices[prices['BTC'] > 50000]
```

### 2.5 기본 통계 계산

```python
# 평균
mean_price = prices['BTC'].mean()

# 표준편차 (변동성)
std_price = prices['BTC'].std()

# 최대/최소
max_price = prices['BTC'].max()
min_price = prices['BTC'].min()

# 한번에 다 보기
print(prices['BTC'].describe())
```

### 2.6 수익률 계산하기

이게 금융 분석의 핵심이에요!

```python
# 일간 수익률 계산
# pct_change() = (오늘 - 어제) / 어제
returns = prices.pct_change()

# 첫 번째 행은 NaN (어제가 없으니까)
# 그래서 보통 제거함
returns = returns.dropna()

# 로그 수익률 (학술 논문에서 많이 씀)
log_returns = np.log(prices / prices.shift(1))
```

**💡 왜 로그 수익률을 쓰나요?**
- 수학적으로 더 좋은 성질이 있어요 (더하기가 가능)
- 정규분포에 더 가까워요
- 논문에서는 로그 수익률을 많이 써요

### 2.7 결측치(NaN) 처리

```python
# NaN 확인
print(prices.isna().sum())  # 각 열별 NaN 개수

# NaN 제거
clean_data = prices.dropna()

# NaN을 0으로 채우기
filled_data = prices.fillna(0)

# NaN을 앞의 값으로 채우기 (forward fill)
filled_data = prices.fillna(method='ffill')

# NaN을 뒤의 값으로 채우기 (backward fill)
filled_data = prices.fillna(method='bfill')
```

### 2.8 이동평균 계산

```python
# 20일 이동평균
ma_20 = prices['BTC'].rolling(window=20).mean()

# 60일 이동평균
ma_60 = prices['BTC'].rolling(window=60).mean()

# 이동 표준편차 (변동성)
rolling_std = prices['BTC'].rolling(window=20).std()
```

**💡 rolling(window=20)이 뭐예요?**
- "20개씩 묶어서"라는 뜻
- 오늘 기준 최근 20일 데이터로 평균 계산
- 매일 한 칸씩 이동하면서 계산 (그래서 "이동"평균)



---

## 3. 금융 데이터 분석 기초

### 3.1 핵심 지표들

금융 논문에서 꼭 나오는 지표들이에요.

```python
import pandas as pd
import numpy as np
from scipy import stats

def calculate_metrics(returns):
    """
    전략의 성과 지표를 계산하는 함수
    
    Parameters:
    - returns: 일간 수익률 Series
    
    Returns:
    - dict: 각종 성과 지표
    """
    # 연간화 (일간 → 연간)
    # 1년 = 약 252 거래일
    ann_return = returns.mean() * 252           # 연간 수익률
    ann_volatility = returns.std() * np.sqrt(252)  # 연간 변동성
    
    # 샤프 비율 = 수익률 / 변동성
    # "위험 대비 수익"을 나타냄
    sharpe_ratio = ann_return / ann_volatility
    
    # 소르티노 비율 = 수익률 / 하락 변동성
    # 손실만 고려한 위험 대비 수익
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino_ratio = ann_return / downside_vol
    
    # 최대 낙폭 (MDD)
    cumulative = (1 + returns).cumprod()  # 누적 수익
    rolling_max = cumulative.cummax()      # 역대 최고점
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # 승률
    win_rate = (returns > 0).mean()
    
    return {
        'ann_return': ann_return,
        'ann_volatility': ann_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }
```

**💡 각 지표 설명:**

| 지표 | 의미 | 좋은 값 |
|------|------|---------|
| 연간 수익률 | 1년에 얼마 버나 | 높을수록 좋음 |
| 변동성 | 얼마나 출렁이나 | 낮을수록 안정적 |
| 샤프 비율 | 위험 대비 수익 | 1 이상이면 좋음, 2 이상이면 훌륭 |
| 소르티노 비율 | 손실 위험 대비 수익 | 높을수록 좋음 |
| MDD | 최악의 손실 | 작을수록 좋음 (보통 음수) |
| 승률 | 돈 번 날 비율 | 50% 이상이면 좋음 |

### 3.2 베타(Beta) 계산

베타는 **시장과 얼마나 같이 움직이나**를 나타내요.

```python
def calculate_beta(asset_returns, market_returns):
    """
    베타 계산
    
    베타 = Cov(자산, 시장) / Var(시장)
    
    - 베타 > 1: 시장보다 더 많이 움직임 (공격적)
    - 베타 = 1: 시장과 똑같이 움직임
    - 베타 < 1: 시장보다 덜 움직임 (방어적)
    - 베타 < 0: 시장과 반대로 움직임
    """
    # 공분산 (같이 움직이는 정도)
    covariance = asset_returns.cov(market_returns)
    
    # 시장 분산
    market_variance = market_returns.var()
    
    # 베타
    beta = covariance / market_variance
    
    return beta

# 사용 예시
btc_beta = calculate_beta(returns['BTC'], market_returns)
print(f"BTC 베타: {btc_beta:.2f}")
```

### 3.3 상관관계 분석

```python
# 두 자산 간 상관관계
correlation = returns['BTC'].corr(returns['ETH'])
print(f"BTC-ETH 상관관계: {correlation:.2f}")

# 모든 자산 간 상관관계 행렬
corr_matrix = returns.corr()
print(corr_matrix)

# 히트맵으로 시각화
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('자산 간 상관관계')
plt.savefig('correlation_heatmap.png')
plt.close()
```

### 3.4 회귀분석 기초

회귀분석은 **"X가 Y에 얼마나 영향을 주나"**를 분석하는 거예요.

```python
from scipy import stats

def simple_regression(y, x):
    """
    단순 회귀분석
    y = alpha + beta * x + error
    
    Parameters:
    - y: 종속변수 (설명하고 싶은 것)
    - x: 독립변수 (설명하는 것)
    
    Returns:
    - alpha: 절편 (x=0일 때 y값)
    - beta: 기울기 (x가 1 증가하면 y가 beta만큼 변화)
    - t_stat: t-통계량 (beta가 0이 아닌지 검정)
    - p_value: p-값 (유의성)
    - r_squared: 설명력 (0~1, 높을수록 잘 설명)
    """
    # scipy의 linregress 사용
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # t-통계량
    t_stat = slope / std_err
    
    return {
        'alpha': intercept,
        'beta': slope,
        't_stat': t_stat,
        'p_value': p_value,
        'r_squared': r_value ** 2
    }

# 사용 예시: BTC 수익률을 시장 수익률로 설명
result = simple_regression(returns['BTC'], market_returns)
print(f"알파: {result['alpha']:.4f}")
print(f"베타: {result['beta']:.2f}")
print(f"t-stat: {result['t_stat']:.2f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"R²: {result['r_squared']:.2f}")
```

**💡 결과 해석:**
- **알파(α)**: 시장과 무관한 초과 수익. 양수면 시장을 이김!
- **베타(β)**: 시장 민감도. 1보다 크면 시장보다 더 출렁임
- **t-stat**: 절대값이 2 이상이면 통계적으로 유의미
- **p-value**: 0.05 미만이면 유의미 (95% 신뢰)
- **R²**: 1에 가까울수록 설명력 높음



---

## 4. 아이디어 → 논문 흐름

### 4.1 금융 논문의 기본 구조

```
1. 서론 (Introduction)
   - 왜 이 연구가 중요한가?
   - 기존 연구의 한계는?
   - 이 논문의 기여는?

2. 문헌 검토 (Literature Review)
   - 관련 선행 연구 정리
   - 이론적 배경

3. 방법론 (Methodology)
   - 데이터 설명
   - 분석 방법 설명
   - 수식 제시

4. 실증 결과 (Empirical Results)
   - 기술 통계량
   - 주요 분석 결과
   - 강건성 검증

5. 결론 (Conclusion)
   - 주요 발견 요약
   - 시사점
   - 한계 및 향후 연구
```

### 4.2 아이디어를 연구 질문으로 바꾸기

**예시: BACBB 전략의 경우**

```
💡 아이디어:
"암호화폐에서도 저베타 자산이 고베타 자산보다 좋지 않을까?"

⬇️

📝 연구 질문:
"암호화폐 시장에서 Betting Against Beta 전략이 유효한가?"

⬇️

🎯 가설:
H1: 저베타 암호화폐 포트폴리오가 고베타 포트폴리오보다 
    위험 조정 수익률이 높다.
H2: BAB 전략의 알파가 통계적으로 유의미하게 양수이다.

⬇️

📊 검증 방법:
1. 베타 계산 → 자산 분류
2. 포트폴리오 구성 → 수익률 계산
3. 성과 비교 → 통계 검정
```

### 4.3 분석 단계별 체크리스트

```python
# ============================================
# 1단계: 데이터 수집 및 정제
# ============================================
"""
□ 데이터 출처 명시 (Yahoo Finance, Binance API 등)
□ 분석 기간 설정
□ 결측치 처리 방법 결정
□ 이상치 처리 방법 결정
□ 수익률 계산 방법 결정 (단순 vs 로그)
"""

# ============================================
# 2단계: 기술 통계량
# ============================================
"""
□ 평균, 표준편차, 최대, 최소
□ 왜도(Skewness), 첨도(Kurtosis)
□ 상관관계
□ 정규성 검정
"""

# ============================================
# 3단계: 핵심 분석
# ============================================
"""
□ 베타/팩터 계산
□ 포트폴리오 구성
□ 수익률 계산
□ 성과 지표 계산
"""

# ============================================
# 4단계: 통계 검정
# ============================================
"""
□ t-검정 (평균이 0과 다른가?)
□ 회귀분석 (알파가 유의한가?)
□ 유의수준 표시 (***, **, *)
"""

# ============================================
# 5단계: 강건성 검증
# ============================================
"""
□ 다른 기간으로 테스트
□ 다른 방법론으로 테스트
□ Out-of-Sample 검증
□ 거래비용 반영
"""
```

### 4.4 코드 구조 설계

좋은 분석 코드는 이런 구조를 가져요:

```python
"""
논문 제목: BACBB 전략 분석
저자: 홍길동
날짜: 2024-01-01
"""

# ============================================
# 0. 라이브러리 임포트
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================
# 1. 설정
# ============================================
DATA_PATH = './data/'
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
RISK_FREE_RATE = 0.05  # 연 5%

# ============================================
# 2. 함수 정의
# ============================================
def load_data():
    """데이터 로드"""
    pass

def calculate_beta():
    """베타 계산"""
    pass

def construct_portfolio():
    """포트폴리오 구성"""
    pass

# ============================================
# 3. 메인 분석
# ============================================
if __name__ == "__main__":
    # 데이터 로드
    data = load_data()
    
    # 분석 실행
    results = analyze(data)
    
    # 결과 저장
    save_results(results)
```



---

## 5. BACBB 코드 한줄한줄 해설

실제 BACBB_Analysis.py 코드를 한 줄씩 설명할게요.

### 5.1 라이브러리 임포트

```python
import pandas as pd          # 데이터 분석용 (엑셀 같은 표 다루기)
import numpy as np           # 수학 계산용 (행렬, 통계 등)
import matplotlib.pyplot as plt  # 그래프 그리기
from scipy import stats      # 통계 검정용
from numpy.linalg import inv # 역행렬 계산용
import warnings
warnings.filterwarnings('ignore')  # 경고 메시지 숨기기
```

**💡 각 라이브러리 역할:**
- `pandas`: CSV 파일 읽기, 데이터 정리, 계산
- `numpy`: 수학 계산 (평균, 표준편차, 행렬 연산)
- `matplotlib`: 그래프 그리기
- `scipy.stats`: t-검정, 회귀분석 등 통계
- `numpy.linalg.inv`: 역행렬 (VAR 모델에서 사용)

### 5.2 한글 폰트 설정

```python
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우용 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지
```

**💡 맥에서는?**
```python
plt.rcParams['font.family'] = 'AppleGothic'
```

### 5.3 데이터 로드

```python
# CSV 파일 읽기
prices = pd.read_csv('04_daily_prices.csv',    # 파일 경로
                     index_col=0,               # 첫 번째 열을 인덱스로
                     parse_dates=True)          # 날짜로 인식

returns = pd.read_csv('06_daily_returns.csv', index_col=0, parse_dates=True)
volumes = pd.read_csv('05_daily_volumes.csv', index_col=0, parse_dates=True)
funding = pd.read_csv('08_daily_funding_rate.csv', index_col=0, parse_dates=True)
```

**💡 각 데이터 설명:**
- `prices`: 일별 가격 (BTC 50000, ETH 3000 등)
- `returns`: 일별 수익률 (0.05 = 5% 상승)
- `volumes`: 일별 거래량
- `funding`: 펀딩비 (선물 거래 비용)

### 5.4 공통 자산 찾기

```python
# 모든 데이터에 공통으로 있는 자산만 선택
common = list(set(prices.columns) & set(returns.columns) & 
              set(volumes.columns) & set(funding.columns))
common = sorted(common)  # 알파벳 순 정렬
print(f"분석 자산: {len(common)}개")
```

**💡 왜 이렇게 하나요?**
- 어떤 데이터에는 BTC가 있고 어떤 데이터에는 없을 수 있어요
- 모든 데이터에 공통으로 있는 자산만 분석해야 오류가 안 나요
- `set()`: 중복 제거, `&`: 교집합

### 5.5 결측치 처리

```python
# 수익률: NaN을 0으로, 극단값 제한
returns = returns.fillna(0).clip(-0.5, 0.5)

# 펀딩비: 앞의 값으로 채우기
funding = funding.fillna(method='ffill').fillna(0)
```

**💡 왜 clip(-0.5, 0.5)?**
- 하루에 -50% ~ +50% 이상 움직이면 데이터 오류일 가능성 높음
- 극단값이 분석을 왜곡할 수 있어서 제한

### 5.6 시장 수익률 계산

```python
# 거래량 가중 평균 수익률 = 시장 수익률
vol_weights = volumes.div(volumes.sum(axis=1), axis=0).fillna(1/len(common))
market_ret = (returns * vol_weights).sum(axis=1)
```

**💡 한 줄씩 설명:**

```python
# 1. 각 자산의 거래량 비중 계산
vol_weights = volumes.div(volumes.sum(axis=1), axis=0)
# volumes.sum(axis=1): 각 날짜별 전체 거래량
# .div(..., axis=0): 각 자산 거래량 / 전체 거래량
# 결과: BTC 0.4, ETH 0.3, ... (합이 1)

# 2. 가중 평균 수익률
market_ret = (returns * vol_weights).sum(axis=1)
# returns * vol_weights: 각 자산 수익률 × 비중
# .sum(axis=1): 날짜별로 합산
# 결과: 시장 전체 수익률
```

### 5.7 VAR 모델 상태변수 구성

```python
# 상태변수 1: 시장 초과수익률
market_excess = market_ret - rf_daily

# 상태변수 2: 기간 스프레드 (장기금리 - 단기금리)
term_spread = (treasury['DGS10'] - treasury['DGS3MO'])

# 상태변수 3: 밸류에이션 (과거 수익률의 음수)
valuation = -prices.mean(axis=1).pct_change(periods=500).fillna(0)
```

**💡 왜 이런 변수들을 쓰나요?**
- **시장 초과수익률**: 현재 시장 상황
- **기간 스프레드**: 경기 전망 (높으면 경기 좋을 것으로 예상)
- **밸류에이션**: 과거에 많이 올랐으면 고평가 (음수로 바꿔서 사용)

### 5.8 VAR 모델로 Cash-Flow News 추출

이 부분이 논문의 핵심이에요!

```python
def estimate_var_and_news(state_df, window=252):
    """
    VAR(1) 모델 추정 및 Cash-Flow News 추출
    
    Campbell-Shiller 분해:
    - 시장 수익률 = 예상 수익률 + 예상치 못한 수익률
    - 예상치 못한 수익률 = Cash-Flow News - Discount Rate News
    """
    
    # 할인율 (일간)
    rho = 0.997  # 연간 약 0.95에 해당
    
    for i in range(window, len(dates)):
        # 윈도우 데이터 (최근 252일)
        z = state_df.iloc[i-window:i].values
        
        # VAR(1) 추정: z_t = c + A * z_{t-1} + u_t
        z_lag = z[:-1]   # 어제 데이터
        z_curr = z[1:]   # 오늘 데이터
        
        # OLS 회귀로 A 행렬 추정
        X = np.column_stack([np.ones(len(z_lag)), z_lag])
        beta = np.linalg.lstsq(X, z_curr, rcond=None)[0]
        A = beta[1:].T
        
        # 잔차 (예상치 못한 부분)
        residuals = z_curr - X @ beta
        u_t = residuals[-1]  # 오늘의 잔차
        
        # Campbell-Shiller 분해
        I = np.eye(n_vars)  # 단위행렬
        inv_term = inv(I - rho * A)  # (I - ρA)^(-1)
        
        # Discount Rate News
        dr_news.iloc[i] = e1 @ (rho * A @ inv_term) @ u_t
        
        # Cash-Flow News = 총 뉴스 + DR News
        cf_news.iloc[i] = u_t[0] + dr_news.iloc[i]
    
    return cf_news, dr_news
```

**💡 쉽게 설명하면:**
1. 과거 데이터로 "내일 시장이 어떨지" 예측하는 모델(VAR)을 만듦
2. 실제 수익률 - 예측 수익률 = 예상치 못한 뉴스
3. 이 뉴스를 두 가지로 분해:
   - **Cash-Flow News**: 회사 실적 같은 영구적 충격
   - **Discount Rate News**: 금리 변화 같은 일시적 충격

### 5.9 Cash-Flow Beta 계산

```python
def estimate_cf_beta(ret_df, cf_news_series, window=60):
    """
    Cash-Flow Beta (Bad Beta) 추정
    
    β_CF = Cov(자산수익률, Cash-Flow News) / Var(Cash-Flow News)
    """
    
    for i in range(window, len(dates)):
        cf_window = cf_arr[i-window:i]      # 최근 60일 CF News
        var_cf = np.var(cf_window)          # CF News의 분산
        
        for j in range(len(cols)):
            r_window = ret_arr[i-window:i, j]  # 자산의 최근 60일 수익률
            
            # 공분산 / 분산 = 베타
            cov_cf = np.cov(r_window, cf_window)[0, 1]
            cf_beta.iloc[i, j] = cov_cf / var_cf
    
    # Shrinkage (극단값 완화)
    cf_beta = cf_beta * 0.6 + 0.4  # 1 방향으로 당기기
    cf_beta = cf_beta.clip(0.1, 3.0)  # 범위 제한
    
    return cf_beta
```

**💡 왜 Shrinkage를 하나요?**
- 베타 추정에는 오차가 있어요
- 극단적인 베타값은 오차일 가능성이 높음
- 1 방향으로 당겨서 오차를 줄임 (Vasicek adjustment)



### 5.10 BACBB 포트폴리오 구성

```python
def construct_bacbb_factor(ret_df, cf_b, fp_b, fund_df, rf, mkt_ret):
    """
    BACBB 팩터 구성
    
    수식: r_BACBB = β_L^(-1) * (r_L - rf) - β_H^(-1) * (r_H - rf)
    
    - Long: Low CF Beta (현금흐름 충격에 방어력)
    - Short: High CF Beta (현금흐름 충격에 취약)
    """
    
    # 비대칭 비중 (롱 70%, 숏 30%)
    LONG_WEIGHT = 0.7
    SHORT_WEIGHT = 0.3
    
    # 매주 리밸런싱
    for week in weeks:
        # 첫 거래일의 베타로 자산 분류
        cfb = cf_b.loc[first_day].dropna()
        
        # 상위/하위 25% 선정
        n_quartile = len(cfb) // 4
        cfb_sorted = cfb.sort_values()
        
        low_cfb = list(cfb_sorted.index[:n_quartile])   # 저베타 (롱)
        high_cfb = list(cfb_sorted.index[-n_quartile:]) # 고베타 (숏)
        
        # 포트폴리오 베타로 레버리지 조절
        beta_L = fp_b[low_cfb].mean()
        beta_H = fp_b[high_cfb].mean()
        
        inv_beta_L = 1.0 / beta_L  # 베타 역수 = 레버리지
        inv_beta_H = 1.0 / beta_H
        
        # 일별 수익률 계산
        for date in week_dates:
            r_long = ret_df.loc[date, low_cfb].mean()   # 롱 포트폴리오 수익률
            r_short = ret_df.loc[date, high_cfb].mean() # 숏 포트폴리오 수익률
            
            # BACBB 수익률
            long_pnl = LONG_WEIGHT * inv_beta_L * (r_long - rf_t)
            short_pnl = SHORT_WEIGHT * inv_beta_H * (-r_short + rf_t)
            
            bacbb_ret.loc[date] = long_pnl + short_pnl
    
    return bacbb_ret
```

**💡 핵심 개념:**

1. **왜 베타 역수로 곱하나요?**
   - 저베타 자산은 덜 움직이니까 더 많이 투자 (레버리지)
   - 고베타 자산은 많이 움직이니까 적게 투자
   - 이렇게 하면 "베타 중립" 포트폴리오가 됨

2. **왜 롱 70%, 숏 30%?**
   - 암호화폐는 전반적으로 상승 추세
   - 숏 비중을 줄여서 상승장에서도 수익

3. **리밸런싱이 뭐예요?**
   - 매주 베타를 다시 계산해서 포트폴리오 재구성
   - 베타는 시간에 따라 변하니까

### 5.11 성과 지표 계산

```python
def calc_metrics(ret, rf, name="Strategy"):
    """전략 성과 지표 계산"""
    
    ret = ret.dropna()
    rf_aligned = rf.reindex(ret.index).fillna(0)
    excess_ret = ret - rf_aligned  # 초과 수익률
    
    # 연간화
    ann_ret = ret.mean() * 252           # 연간 수익률
    ann_vol = ret.std() * np.sqrt(252)   # 연간 변동성
    
    # 샤프 비율
    sharpe = ann_ret / ann_vol
    
    # 소르티노 비율 (하락 변동성만 고려)
    downside = ret[ret < 0]
    downside_vol = downside.std() * np.sqrt(252)
    sortino = ann_ret / downside_vol
    
    # 누적 수익률
    cum_ret = (1 + ret).cumprod()
    total_ret = cum_ret.iloc[-1] - 1
    
    # 최대 낙폭 (MDD)
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    mdd = drawdown.min()
    
    # 칼마 비율 (수익률 / MDD)
    calmar = ann_ret / abs(mdd)
    
    # 승률
    win_rate = (ret > 0).mean()
    
    # t-검정 (수익률이 0과 다른가?)
    n = len(ret)
    mean_ret = ret.mean()
    se = ret.std() / np.sqrt(n)  # 표준오차
    t_stat = mean_ret / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-1))
    
    return {
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'total_ret': total_ret,
        'mdd': mdd,
        'win_rate': win_rate,
        't_stat': t_stat,
        'p_value': p_value
    }
```

**💡 각 지표 해석:**

| 지표 | 계산 | 의미 |
|------|------|------|
| 연간 수익률 | 일평균 × 252 | 1년 기대 수익 |
| 연간 변동성 | 일표준편차 × √252 | 1년 기대 변동폭 |
| 샤프 비율 | 수익률 / 변동성 | 위험 대비 수익 |
| 소르티노 | 수익률 / 하락변동성 | 손실 위험 대비 수익 |
| MDD | 최고점 대비 최대 하락 | 최악의 손실 |
| 칼마 비율 | 수익률 / MDD | MDD 대비 수익 |
| t-stat | 평균 / 표준오차 | 통계적 유의성 |

### 5.12 5분위 분석

```python
def quintile_analysis(ret_df, beta_df, name="Beta"):
    """
    베타 기준 5분위 포트폴리오 분석
    
    Q1: 베타 가장 낮은 20%
    Q2: 베타 낮은 20%
    Q3: 베타 중간 20%
    Q4: 베타 높은 20%
    Q5: 베타 가장 높은 20%
    """
    
    quintile_returns = {f'Q{i}': [] for i in range(1, 6)}
    
    for month in months:
        # 월초 베타로 분류
        b = beta_df.loc[first_day].dropna()
        b_sorted = b.sort_values()
        
        # 5등분
        quintiles = np.array_split(b_sorted.index, 5)
        
        # 각 분위별 수익률 계산
        for date in month_dates:
            for i, q_assets in enumerate(quintiles):
                q_ret = ret_df.loc[date, q_assets].mean()
                quintile_returns[f'Q{i+1}'].append(q_ret)
    
    # Q1-Q5 스프레드 (저베타 - 고베타)
    spread = Q1_returns - Q5_returns
    
    return results
```

**💡 왜 5분위 분석을 하나요?**
- "저베타가 정말 좋은가?"를 검증
- Q1(저베타)이 Q5(고베타)보다 수익률이 높으면 가설 지지
- Q1-Q5 스프레드가 양수이고 유의하면 전략이 유효



---

## 6. 통계 검정 방법

### 6.1 t-검정 (평균이 0과 다른가?)

```python
from scipy import stats

def t_test(returns):
    """
    단일 표본 t-검정
    
    H0 (귀무가설): 평균 수익률 = 0 (전략이 효과 없음)
    H1 (대립가설): 평균 수익률 ≠ 0 (전략이 효과 있음)
    """
    n = len(returns)
    mean = returns.mean()
    std = returns.std()
    se = std / np.sqrt(n)  # 표준오차
    
    # t-통계량
    t_stat = mean / se
    
    # p-값 (양측 검정)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-1))
    
    return t_stat, p_value

# 사용 예시
t, p = t_test(bacbb_returns)
print(f"t-stat: {t:.2f}")
print(f"p-value: {p:.4f}")

# 해석
if p < 0.01:
    print("*** 1% 수준에서 유의 (매우 강한 증거)")
elif p < 0.05:
    print("** 5% 수준에서 유의 (강한 증거)")
elif p < 0.10:
    print("* 10% 수준에서 유의 (약한 증거)")
else:
    print("유의하지 않음")
```

**💡 쉽게 설명:**
- t-stat 절대값이 2 이상이면 대략 유의
- p-value가 0.05 미만이면 "우연이 아니다"라고 95% 확신

### 6.2 회귀분석 (알파 검정)

```python
from scipy import stats

def alpha_test(strategy_returns, market_returns, rf):
    """
    CAPM 회귀분석으로 알파 검정
    
    r_strategy - rf = alpha + beta * (r_market - rf) + error
    
    alpha > 0이고 유의하면: 시장을 이기는 전략!
    """
    # 초과 수익률
    excess_strategy = strategy_returns - rf
    excess_market = market_returns - rf
    
    # 회귀분석
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        excess_market, excess_strategy
    )
    
    # 알파의 t-통계량
    n = len(excess_strategy)
    residuals = excess_strategy - (intercept + slope * excess_market)
    mse = (residuals ** 2).sum() / (n - 2)
    se_alpha = np.sqrt(mse / n)
    t_alpha = intercept / se_alpha
    p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), n-2))
    
    return {
        'alpha': intercept * 252,  # 연간화
        'beta': slope,
        't_stat': t_alpha,
        'p_value': p_alpha,
        'r_squared': r_value ** 2
    }

# 사용 예시
result = alpha_test(bacbb_returns, market_returns, rf_daily)
print(f"연간 알파: {result['alpha']*100:.2f}%")
print(f"베타: {result['beta']:.2f}")
print(f"t-stat: {result['t_stat']:.2f}")
print(f"p-value: {result['p_value']:.4f}")
```

### 6.3 유의수준 별표 표시

논문에서 흔히 쓰는 표기법:

```python
def get_significance_stars(p_value):
    """
    p-value에 따른 유의수준 별표
    
    ***: p < 0.01 (99% 신뢰)
    **:  p < 0.05 (95% 신뢰)
    *:   p < 0.10 (90% 신뢰)
    """
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.10:
        return "*"
    else:
        return ""

# 사용 예시
stars = get_significance_stars(0.023)  # "**"
print(f"수익률: 15.2%{stars}")  # "수익률: 15.2%**"
```

### 6.4 Newey-West 표준오차

시계열 데이터는 자기상관이 있어서 일반 표준오차가 부정확해요.

```python
def newey_west_tstat(returns, lags=5):
    """
    Newey-West 조정 t-통계량
    
    자기상관을 고려한 더 정확한 표준오차
    """
    n = len(returns)
    mean = returns.mean()
    
    # 자기공분산 계산
    gamma = []
    for j in range(lags + 1):
        if j == 0:
            gamma.append(((returns - mean) ** 2).sum() / n)
        else:
            gamma.append(((returns[j:] - mean) * (returns[:-j] - mean)).sum() / n)
    
    # Newey-West 분산
    nw_var = gamma[0]
    for j in range(1, lags + 1):
        weight = 1 - j / (lags + 1)  # Bartlett 가중치
        nw_var += 2 * weight * gamma[j]
    
    # 표준오차
    nw_se = np.sqrt(nw_var / n)
    
    # t-통계량
    t_stat = mean / nw_se
    
    return t_stat

# 사용 예시
t_nw = newey_west_tstat(bacbb_returns)
print(f"Newey-West t-stat: {t_nw:.2f}")
```

### 6.5 Out-of-Sample 검증

과적합(overfitting)을 방지하기 위한 검증:

```python
def out_of_sample_test(returns):
    """
    In-Sample / Out-of-Sample 분리 검증
    
    - 전반부로 전략 개발 (In-Sample)
    - 후반부로 검증 (Out-of-Sample)
    """
    # 데이터 반으로 나누기
    split_idx = len(returns) // 2
    
    in_sample = returns.iloc[:split_idx]
    out_of_sample = returns.iloc[split_idx:]
    
    # 각각 성과 계산
    is_sharpe = in_sample.mean() / in_sample.std() * np.sqrt(252)
    oos_sharpe = out_of_sample.mean() / out_of_sample.std() * np.sqrt(252)
    
    print(f"In-Sample 샤프: {is_sharpe:.2f}")
    print(f"Out-of-Sample 샤프: {oos_sharpe:.2f}")
    
    # OOS 샤프가 IS의 50% 이상이면 양호
    ratio = oos_sharpe / is_sharpe if is_sharpe != 0 else 0
    print(f"OOS/IS 비율: {ratio:.1%}")
    
    return is_sharpe, oos_sharpe

# 사용 예시
is_s, oos_s = out_of_sample_test(bacbb_returns)
```

**💡 왜 중요한가요?**
- In-Sample에서만 좋으면 과적합일 수 있음
- Out-of-Sample에서도 좋아야 진짜 유효한 전략
- OOS 샤프가 IS의 50% 이상이면 괜찮은 편



---

## 7. 그래프와 표 만들기

### 7.1 기본 그래프 설정

```python
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
# plt.rcParams['font.family'] = 'AppleGothic'  # 맥
plt.rcParams['axes.unicode_minus'] = False

# 기본 스타일 설정
plt.rcParams['figure.figsize'] = (12, 6)  # 그래프 크기
plt.rcParams['font.size'] = 12            # 글자 크기
```

### 7.2 누적 수익률 그래프

```python
def plot_cumulative_returns(returns_dict, title="누적 수익률"):
    """
    여러 전략의 누적 수익률 비교 그래프
    
    Parameters:
    - returns_dict: {'전략명': 수익률Series} 형태
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, returns) in enumerate(returns_dict.items()):
        # 누적 수익률 계산
        cumulative = (1 + returns).cumprod()
        
        # 그래프 그리기
        ax.plot(cumulative.index, cumulative.values, 
                label=name, color=colors[i % len(colors)], linewidth=1.5)
    
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)  # 기준선
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('날짜')
    ax.set_ylabel('누적 수익률')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cumulative_returns.png', dpi=150)
    plt.close()

# 사용 예시
plot_cumulative_returns({
    'BACBB': bacbb_returns,
    'BACB': bacb_returns,
    'Buy & Hold': bh_returns
})
```

### 7.3 드로우다운 그래프

```python
def plot_drawdown(returns, title="드로우다운"):
    """최대 낙폭 시각화"""
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 누적 수익률
    cumulative = (1 + returns).cumprod()
    
    # 드로우다운 계산
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max * 100
    
    # 그래프
    ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown, color='red', linewidth=1)
    
    ax.set_title(f'{title} (MDD: {drawdown.min():.1f}%)', fontsize=14)
    ax.set_xlabel('날짜')
    ax.set_ylabel('드로우다운 (%)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('drawdown.png', dpi=150)
    plt.close()

# 사용 예시
plot_drawdown(bacbb_returns, "BACBB 드로우다운")
```

### 7.4 연도별 수익률 막대 그래프

```python
def plot_yearly_returns(returns, title="연도별 수익률"):
    """연도별 수익률 막대 그래프"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 연도별 수익률 계산
    yearly = returns.groupby(returns.index.year).apply(
        lambda x: (1 + x).prod() - 1
    ) * 100
    
    # 색상 (양수=초록, 음수=빨강)
    colors = ['green' if r > 0 else 'red' for r in yearly]
    
    # 막대 그래프
    ax.bar(yearly.index.astype(str), yearly.values, 
           color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('연도')
    ax.set_ylabel('수익률 (%)')
    
    # 값 표시
    for i, (year, ret) in enumerate(yearly.items()):
        ax.text(i, ret + (2 if ret > 0 else -4), f'{ret:.1f}%', 
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('yearly_returns.png', dpi=150)
    plt.close()

# 사용 예시
plot_yearly_returns(bacbb_returns, "BACBB 연도별 수익률")
```

### 7.5 수익률 분포 히스토그램

```python
def plot_distribution(returns, title="수익률 분포"):
    """수익률 분포 히스토그램"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 히스토그램
    ax.hist(returns * 100, bins=50, alpha=0.7, 
            color='blue', edgecolor='black', density=True)
    
    # 평균선
    mean_ret = returns.mean() * 100
    ax.axvline(x=mean_ret, color='red', linestyle='--', 
               linewidth=2, label=f'평균: {mean_ret:.3f}%')
    
    # 0 기준선
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('일간 수익률 (%)')
    ax.set_ylabel('빈도')
    ax.legend()
    
    # 통계 정보 추가
    stats_text = f"""
    평균: {returns.mean()*100:.3f}%
    표준편차: {returns.std()*100:.2f}%
    왜도: {returns.skew():.2f}
    첨도: {returns.kurtosis():.2f}
    """
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('distribution.png', dpi=150)
    plt.close()

# 사용 예시
plot_distribution(bacbb_returns, "BACBB 수익률 분포")
```

### 7.6 논문용 표 만들기 (LaTeX)

```python
def create_latex_table(data, caption, label):
    """
    논문용 LaTeX 표 생성
    
    Parameters:
    - data: DataFrame
    - caption: 표 제목
    - label: 참조용 라벨
    """
    
    latex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{'l' + 'c' * (len(data.columns))}}}
\\toprule
"""
    
    # 헤더
    latex += " & ".join([""] + list(data.columns)) + " \\\\\n"
    latex += "\\midrule\n"
    
    # 데이터
    for idx, row in data.iterrows():
        values = [str(idx)]
        for val in row:
            if isinstance(val, float):
                values.append(f"{val:.2f}")
            else:
                values.append(str(val))
        latex += " & ".join(values) + " \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    return latex

# 사용 예시
performance_df = pd.DataFrame({
    'Ann. Return': [15.2, 12.1, 8.5],
    'Volatility': [18.3, 22.1, 35.2],
    'Sharpe': [0.83, 0.55, 0.24]
}, index=['BACBB', 'BACB', 'B&H'])

latex_code = create_latex_table(
    performance_df, 
    "Strategy Performance Comparison",
    "tab:performance"
)
print(latex_code)
```

### 7.7 HTML 표 만들기 (브라우저용)

```python
def create_html_table(data, title):
    """브라우저에서 볼 수 있는 HTML 표"""
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #ddd; }}
        h2 {{ color: #333; }}
    </style>
</head>
<body>
    <h2>{title}</h2>
    {data.to_html(classes='styled-table')}
</body>
</html>
"""
    
    with open('table.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print("table.html 저장 완료!")

# 사용 예시
create_html_table(performance_df, "전략 성과 비교")
```



---

## 8. 실전 팁과 자주 하는 실수

### 8.1 자주 하는 실수들

#### ❌ 실수 1: Look-Ahead Bias (미래 정보 사용)

```python
# ❌ 잘못된 코드: 미래 데이터로 베타 계산
beta = returns.rolling(60).apply(lambda x: x.cov(market) / market.var())
# 문제: rolling은 현재 시점 포함! 미래를 본 것

# ✅ 올바른 코드: shift로 하루 밀기
beta = returns.shift(1).rolling(60).apply(...)
# 어제까지의 데이터로만 계산
```

#### ❌ 실수 2: Survivorship Bias (생존자 편향)

```python
# ❌ 잘못된 예: 현재 존재하는 코인만 분석
# 상장폐지된 코인은 보통 성과가 나빴음
# 이걸 빼면 성과가 과대평가됨

# ✅ 올바른 방법: 상장폐지 코인도 포함
# 데이터 수집 시 과거 모든 코인 포함
```

#### ❌ 실수 3: 거래비용 무시

```python
# ❌ 잘못된 코드: 거래비용 없이 계산
portfolio_return = long_return - short_return

# ✅ 올바른 코드: 거래비용 반영
TRADING_FEE = 0.0004  # 0.04%
turnover = calculate_turnover()  # 회전율
trading_cost = turnover * TRADING_FEE * 2  # 매수+매도
portfolio_return = long_return - short_return - trading_cost
```

#### ❌ 실수 4: 데이터 스누핑 (과적합)

```python
# ❌ 잘못된 방법: 전체 데이터로 파라미터 최적화
best_window = optimize_window(all_data)  # 전체 데이터 사용

# ✅ 올바른 방법: In-Sample로만 최적화
train_data = all_data[:len(all_data)//2]
test_data = all_data[len(all_data)//2:]

best_window = optimize_window(train_data)  # 훈련 데이터만
test_performance = evaluate(test_data, best_window)  # 테스트로 검증
```

### 8.2 디버깅 팁

```python
# 1. 데이터 확인
print(df.head())        # 처음 5줄
print(df.tail())        # 마지막 5줄
print(df.shape)         # (행, 열) 크기
print(df.info())        # 데이터 타입, 결측치
print(df.describe())    # 기본 통계

# 2. 결측치 확인
print(df.isna().sum())  # 열별 결측치 개수
print(df.isna().any())  # 결측치 있는 열

# 3. 중간 결과 출력
def calculate_something(data):
    step1 = data * 2
    print(f"Step 1 결과: {step1.head()}")  # 중간 확인
    
    step2 = step1.rolling(20).mean()
    print(f"Step 2 결과: {step2.head()}")  # 중간 확인
    
    return step2

# 4. assert로 가정 확인
assert len(returns) == len(prices) - 1, "수익률 길이 오류!"
assert returns.isna().sum().sum() == 0, "결측치 있음!"
assert (weights.sum(axis=1) - 1).abs().max() < 0.01, "비중 합이 1이 아님!"
```

### 8.3 코드 정리 팁

```python
# 1. 상수는 맨 위에 대문자로
TRADING_FEE = 0.0004
RISK_FREE_RATE = 0.05
REBALANCE_FREQ = 'W'  # Weekly

# 2. 함수에 docstring 쓰기
def calculate_sharpe(returns, rf=0):
    """
    샤프 비율 계산
    
    Parameters:
    -----------
    returns : pd.Series
        일간 수익률
    rf : float
        무위험 수익률 (연간, 기본값 0)
    
    Returns:
    --------
    float
        연간화된 샤프 비율
    """
    excess = returns - rf/252
    return excess.mean() / excess.std() * np.sqrt(252)

# 3. 의미 있는 변수명
# ❌ 나쁜 예
x = df['BTC'].pct_change()
y = x.rolling(20).std()

# ✅ 좋은 예
btc_returns = df['BTC'].pct_change()
btc_volatility = btc_returns.rolling(20).std()
```

### 8.4 성능 최적화 팁

```python
# 1. 벡터 연산 사용 (for 루프 피하기)
# ❌ 느린 코드
result = []
for i in range(len(df)):
    result.append(df.iloc[i]['A'] * df.iloc[i]['B'])

# ✅ 빠른 코드
result = df['A'] * df['B']

# 2. apply 대신 내장 함수
# ❌ 느린 코드
df['return'] = df['price'].apply(lambda x: x / df['price'].shift(1) - 1)

# ✅ 빠른 코드
df['return'] = df['price'].pct_change()

# 3. 큰 데이터는 청크로 처리
# 메모리 부족할 때
for chunk in pd.read_csv('big_file.csv', chunksize=10000):
    process(chunk)
```

### 8.5 논문 작성 체크리스트

```
□ 데이터
  □ 출처 명시
  □ 기간 명시
  □ 결측치 처리 방법 설명
  □ 이상치 처리 방법 설명

□ 방법론
  □ 수식 제시
  □ 파라미터 설명
  □ 리밸런싱 주기 명시
  □ 거래비용 반영 여부

□ 결과
  □ 기술 통계량 표
  □ 주요 결과 표
  □ 유의수준 표시 (***, **, *)
  □ 그래프 (누적수익률, 드로우다운 등)

□ 강건성
  □ 다른 기간 테스트
  □ Out-of-Sample 검증
  □ 다른 파라미터 테스트
  □ 거래비용 민감도 분석

□ 코드
  □ 재현 가능하게 정리
  □ 주석 달기
  □ README 작성
```

---

## 🎯 마무리

이 가이드를 따라하면서 중요한 것:

1. **천천히, 한 줄씩** - 급하게 하면 오류 남
2. **중간중간 확인** - print()로 결과 확인
3. **에러 메시지 읽기** - 구글에 검색하면 답 나옴
4. **작은 것부터** - 간단한 분석부터 시작

처음엔 어렵지만, 몇 번 해보면 패턴이 보여요. 화이팅! 🚀

---

## 📖 추가 학습 자료

- **파이썬 기초**: [점프 투 파이썬](https://wikidocs.net/book/1)
- **pandas**: [10 Minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- **금융 분석**: [QuantStart](https://www.quantstart.com/)
- **통계**: [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability)

---

*이 가이드는 BACBB 프로젝트를 기반으로 작성되었습니다.*
*질문이 있으면 코드와 함께 구체적으로 물어보세요!*
