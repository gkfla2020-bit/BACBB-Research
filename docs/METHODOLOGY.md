# Methodology: BACBB Strategy

## 1. Theoretical Framework

### 1.1 Low Beta Anomaly

Black(1972)은 CAPM의 Security Market Line이 이론보다 평평함을 발견했다. 이는 저베타 자산이 CAPM 예측보다 높은 수익률을, 고베타 자산이 낮은 수익률을 제공함을 의미한다.

Frazzini & Pedersen(2014)은 이 현상을 **레버리지 제약**으로 설명했다:
- 레버리지 제약이 있는 투자자들은 높은 기대수익률을 위해 고베타 자산을 선호
- 이로 인해 고베타 자산이 과대평가되고 저베타 자산이 과소평가됨

### 1.2 Bad Beta Concept

Campbell & Vuolteenaho(2004)는 CAPM Beta를 두 가지 구성요소로 분해했다:

```
β = β_CF + β_DR
```

| Component | Description | Risk Type |
|-----------|-------------|-----------|
| **β_CF** (Cash-Flow Beta) | 영구적 현금흐름 충격에 대한 민감도 | **Bad Beta** |
| **β_DR** (Discount Rate Beta) | 일시적 할인율 변동에 대한 민감도 | Good Beta |

**Cash-Flow News**는 기업의 펀더멘털에 대한 영구적 충격을 반영한다. 따라서 β_CF가 높은 자산은 펀더멘털 악화에 취약하며, 이를 "나쁜 베타"라 한다.

---

## 2. VAR Model Specification

### 2.1 State Variables

| Variable | Definition | Rationale |
|----------|------------|-----------|
| z₁ | Market Excess Return | 시가총액 가중 평균 - 무위험수익률 |
| z₂ | Term Spread | 10년 국채 - 3개월 국채 |
| z₃ | Valuation Indicator | 과거 100주 누적수익률의 음수 |

### 2.2 VAR(1) Model

```
z_{t+1} = c + A · z_t + u_{t+1}
```

- **Estimation:** Rolling Window OLS
- **Window Size:** 252 trading days
- **Update Frequency:** Daily

### 2.3 Campbell-Shiller Decomposition

**Discount Factor:** ρ = 0.997 (daily)

**Cash-Flow News:**
```
N_CF = (e₁' + e₁' · ρ · A · (I - ρA)⁻¹) · u_{t+1}
```

**Discount Rate News:**
```
N_DR = e₁' · ρ · A · (I - ρA)⁻¹ · u_{t+1}
```

여기서 e₁은 첫 번째 변수 선택 벡터 [1, 0, 0]'이다.

---

## 3. Cash-Flow Beta Estimation

### 3.1 Definition

```
β_CF,i = Cov(r_i, N_CF) / Var(N_CF)
```

### 3.2 Estimation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Rolling Window | 60 days | Beta 추정 윈도우 |
| Update Frequency | 5 days | 리밸런싱 주기 |
| Vasicek Adjustment | 0.6 × β_raw + 0.4 | 극단값 완화 |
| Clipping Range | [0.1, 3.0] | 극단값 제한 |

---

## 4. Portfolio Construction

### 4.1 Asset Selection

| Position | Selection Criteria | Rationale |
|----------|-------------------|-----------|
| **Long** | Bottom 25% CF Beta | 현금흐름 충격 방어력 |
| **Short** | Top 25% CF Beta | 현금흐름 충격 취약 |

### 4.2 Position Sizing

```
r_BACBB = w_L · β_L⁻¹ · (r_L - r_f - f) - w_S · β_H⁻¹ · (r_H - r_f - f)
```

| Parameter | Value |
|-----------|-------|
| Long Weight (w_L) | 70% |
| Short Weight (w_S) | 30% |
| Max Leverage | 2.0 |
| Min Leverage | 0.5 |

### 4.3 Volatility Scaling

- **Target Volatility:** 22% annualized
- **Method:** Rolling volatility-based dynamic adjustment

---

## 5. Transaction Costs

| Cost Type | Rate | Frequency |
|-----------|------|-----------|
| Binance Taker Fee | 0.04% | Per trade |
| Funding Rate | Actual data | 8-hour intervals |
| Rebalancing Cost | Turnover × 0.04% × 2 | Weekly |

---

## 6. Statistical Testing

### 6.1 Significance Tests

| Test | Statistic | Interpretation |
|------|-----------|----------------|
| t-test | t = 2.79 | Mean return ≠ 0 |
| p-value | 0.0054 | 1% significance level |

### 6.2 Out-of-Sample Validation

- **Split Point:** 50% of data (2023.07)
- **In-Sample:** Model estimation
- **Out-of-Sample:** Strategy validation

---

## References

1. Campbell, J. Y., & Vuolteenaho, T. (2004). Bad beta, good beta. *American Economic Review*, 94(5), 1249-1275.

2. Frazzini, A., & Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1-25.
