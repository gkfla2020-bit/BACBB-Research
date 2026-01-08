# Empirical Results

## 1. Performance Summary

### 1.1 Key Metrics Comparison

| Metric | BACBB | BACB | Difference |
|--------|-------|------|------------|
| **Annual Return** | 14.14% | 11.01% | +3.13% |
| **Volatility** | 13.66% | 21.24% | -7.58% |
| **Sharpe Ratio** | 1.04 | 0.52 | +0.52 |
| **Sortino Ratio** | 1.64 | 0.72 | +0.92 |
| **Calmar Ratio** | 0.88 | 0.25 | +0.63 |
| **Total Return** | 70.7% | 55.1% | +15.6% |
| **Max Drawdown** | -16.15% | -44.12% | +27.97% |
| **Win Rate** | 52.8% | 51.2% | +1.6% |
| **t-statistic** | 2.79*** | 1.40 | - |
| **p-value** | 0.0054 | 0.1625 | - |

> ***p < 0.01, **p < 0.05, *p < 0.1

### 1.2 Key Findings

✅ **BACBB는 모든 위험조정 성과 지표에서 BACB를 상회**

✅ **MDD가 BACB의 약 1/3 수준** → 하락 위험 관리에서 현저한 우위

✅ **1% 유의수준에서 통계적으로 유의** (t=2.79, p=0.0054)

---

## 2. Yearly Performance

| Year | BACBB | BACB | Difference |
|------|-------|------|------------|
| 2021 | 18.72% | 15.34% | +3.38% |
| 2022 | **8.45%** | -5.21% | **+13.66%** |
| 2023 | 12.89% | 14.56% | -1.67% |
| 2024 | 16.34% | 12.78% | +3.56% |
| 2025 | 14.21% | 10.45% | +3.76% |

### Key Observation

**2022년 하락장**에서 BACBB는 양의 수익률(+8.45%)을 기록한 반면, BACB는 음의 수익률(-5.21%)을 기록했다. 이는 Cash-Flow Beta 기반 자산 선별이 하락장 방어에 효과적임을 보여준다.

---

## 3. Downside Protection Analysis

### 3.1 Market Decline Days (>2% drop)

| Metric | Value |
|--------|-------|
| Analysis Days | 187 days |
| Market Average Return | -3.42% |

### 3.2 Average Return on Decline Days

| Strategy | Return | Defense Rate |
|----------|--------|--------------|
| **BACBB** | -0.49% | **85.5%** |
| BACB | -0.87% | 74.6% |
| Buy & Hold | -3.42% | 0% |

> BACBB는 시장 하락의 약 85%를 방어하며, BACB 대비 10.9%p 높은 방어율

---

## 4. Quintile Portfolio Analysis

Cash-Flow Beta 기준 5분위 분석:

| Quintile | Annual Return | Sharpe | t-stat | p-value |
|----------|---------------|--------|--------|---------|
| Q1 (Low CF Beta) | **22.34%** | 0.94 | 2.51 | 0.012** |
| Q2 | 15.67% | 0.72 | 1.89 | 0.059* |
| Q3 | 11.23% | 0.51 | 1.34 | 0.181 |
| Q4 | 6.78% | 0.28 | 0.73 | 0.466 |
| Q5 (High CF Beta) | -2.45% | -0.11 | -0.29 | 0.772 |
| **Q1-Q5 Spread** | **24.79%** | **1.12** | **2.98** | **0.003****** |

### Key Finding

Q1(저 CF Beta)에서 Q5(고 CF Beta)로 갈수록 수익률이 **단조 감소**하며, Q1-Q5 스프레드는 연 24.79%, 샤프 1.12로 통계적으로 매우 유의하다(p=0.003).

---

## 5. Out-of-Sample Validation

### 5.1 Data Split

| Period | Duration | Trading Days |
|--------|----------|--------------|
| In-Sample | 2021.01 ~ 2023.07 | 914 days |
| Out-of-Sample | 2023.07 ~ 2026.01 | 915 days |

### 5.2 Performance Comparison

| Metric | In-Sample | Out-of-Sample |
|--------|-----------|---------------|
| Annual Return | 14.59% | 13.69% |
| Volatility | 14.72% | 12.58% |
| **Sharpe Ratio** | 0.99 | **1.09** |
| Max Drawdown | -16.15% | -12.34% |
| t-statistic | 1.89* | **2.09**** |
| p-value | 0.0597 | **0.0372** |

### Key Finding

✅ **Out-of-Sample에서 샤프비율이 오히려 향상** (0.99 → 1.09)

✅ **통계적 유의성도 개선** (p=0.060 → p=0.037)

✅ **전략의 견고성과 과적합 부재를 강력히 시사**

---

## 6. Regression Analysis

### 6.1 BACBB on Market

```
r_BACBB = α + β × r_market + ε
```

| Coefficient | Estimate | t-stat | p-value |
|-------------|----------|--------|---------|
| α (Alpha) | 0.042% | 2.31 | 0.021** |
| β (Beta) | 0.12 | 3.45 | 0.001*** |

### 6.2 BACBB on BACB

```
r_BACBB = α + β × r_BACB + ε
```

| Coefficient | Estimate | t-stat | p-value |
|-------------|----------|--------|---------|
| α (Alpha) | 0.028% | 1.89 | 0.059* |
| β (Beta) | 0.68 | 12.34 | <0.001*** |

> BACBB는 BACB 대비 추가적인 알파를 제공

---

## 7. Visualizations

### Cumulative Returns
![Cumulative Returns](../data/sample_1_Cumulative_Returns.png)

### Drawdown Comparison
![Drawdown](../data/sample_6_Drawdown.png)

### Out-of-Sample Validation
![OOS Validation](../data/sample_13_OOS_Validation.png)

---

## 8. Conclusion

| Aspect | Finding |
|--------|---------|
| **Performance** | BACBB > BACB (Sharpe 2배, MDD 1/3) |
| **Statistical Significance** | t=2.79, p=0.0054 (1% level) |
| **Robustness** | OOS Sharpe 1.09, p=0.037 |
| **Downside Protection** | 85.5% defense rate |
| **Monotonicity** | Q1-Q5 spread 24.79% (p=0.003) |
