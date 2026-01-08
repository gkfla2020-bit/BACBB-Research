# BACBB: Betting Against Cryptocurrency Bad Beta

<div align="center">

**VAR 모델 기반 Cash-Flow Beta를 활용한 암호화폐 팩터 투자 전략**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Complete-brightgreen.svg)]()

**Author:** [gkfla2020-bit](https://github.com/gkfla2020-bit)

[📄 Full Report](docs/BACBB_Full_Report.html) · [📊 Analysis Code](data/BACBB_Analysis.py) · [📈 Results](#results)

</div>

---

## Abstract

본 연구는 **Frazzini & Pedersen(2014)**의 BAB(Betting Against Beta) 전략과 **Campbell & Vuolteenaho(2004)**의 Bad Beta 개념을 암호화폐 시장에 적용한 **BACBB(Betting Against Cryptocurrency Bad Beta)** 전략을 제안한다.

VAR(Vector Autoregression) 모델과 Campbell-Shiller 분해를 통해 시장 수익률을 **Cash-Flow News**와 **Discount Rate News**로 분해하고, 각 자산의 **Cash-Flow Beta(β_CF)**를 추정하여 "진정한 나쁜 베타"를 식별한다.

### Key Results

| Metric | BACBB | BACB | Improvement |
|--------|-------|------|-------------|
| **Annual Return** | 14.14% | 11.01% | +3.13% |
| **Sharpe Ratio** | 1.04 | 0.52 | **+100%** |
| **Max Drawdown** | -16.15% | -44.12% | **+63%** |
| **t-statistic** | 2.79*** | 1.40 | - |

> ***p < 0.01** — 1% 유의수준에서 통계적으로 유의

---

## Research Highlights

### 🎯 Core Innovation

기존 BACB 전략은 **Total Beta**만을 고려하여 일시적 할인율 변동과 영구적 현금흐름 충격을 구분하지 못하는 한계가 있다. 본 연구는 VAR 모델을 통해 **Cash-Flow Beta**를 추출하여 "진정한 나쁜 베타"를 식별한다.

```
β = β_CF + β_DR

β_CF (Cash-Flow Beta): 영구적 현금흐름 충격에 대한 민감도 → "Bad Beta"
β_DR (Discount Rate Beta): 일시적 할인율 변동에 대한 민감도 → "Good Beta"
```

### 📊 Methodology

**VAR(1) Model:**
```
z_{t+1} = c + A · z_t + u_{t+1}
```

**State Variables:**
- z₁: Market Excess Return
- z₂: Term Spread (10Y - 3M Treasury)
- z₃: Valuation Indicator

**Cash-Flow News Extraction:**
```
N_CF = (e₁' + e₁' · ρ · A · (I - ρA)⁻¹) · u_{t+1}
```

### 📈 Portfolio Construction

| Parameter | Value |
|-----------|-------|
| Long Position | Low CF Beta (Bottom 25%) |
| Short Position | High CF Beta (Top 25%) |
| Long Weight | 70% |
| Short Weight | 30% |
| Transaction Cost | 0.04% (Binance Taker) |
| Rebalancing | Weekly |

---

## Results

### Cumulative Returns (2021-2026)

<p align="center">
  <img src="data/sample_1_Cumulative_Returns.png" width="80%" alt="Cumulative Returns">
</p>

BACBB 전략은 전 기간에 걸쳐 BACB 대비 안정적인 누적 수익률을 기록했으며, 특히 **2022년 하락장에서 방어력**이 두드러진다.

### Drawdown Comparison

<p align="center">
  <img src="data/sample_6_Drawdown.png" width="80%" alt="Drawdown">
</p>

BACBB의 MDD(-16.15%)는 BACB(-44.12%)의 **약 1/3 수준**으로, 하락 위험 관리에서 현저한 우위를 보인다.

### Out-of-Sample Validation

| Period | Duration | Annual Return | Sharpe | p-value |
|--------|----------|---------------|--------|---------|
| In-Sample | 2021.01 ~ 2023.07 | 14.59% | 0.99 | 0.060* |
| **Out-of-Sample** | 2023.07 ~ 2026.01 | **13.69%** | **1.09** | **0.037***** |

> OOS에서 샤프비율이 오히려 **향상**(0.99 → 1.09)되어 전략의 견고성이 확인됨

---

## Data & Analysis

### Dataset

- **Assets:** 50 cryptocurrencies (Binance)
- **Period:** 2021.01.01 ~ 2026.01.05 (1,829 trading days)
- **Data Sources:**
  - Daily prices, volumes, returns
  - Funding rates (8-hour intervals)
  - US Treasury rates (3M, 10Y)

### File Structure

```
data/
├── 01_crypto_prices_raw.csv      # Raw price data
├── 02_treasury_rates.csv         # US Treasury rates
├── 03_funding_rates_raw.csv      # Funding rate data
├── 04_daily_prices.csv           # Processed daily prices
├── 05_daily_volumes.csv          # Daily trading volumes
├── 06_daily_returns.csv          # Daily returns
├── 07_daily_log_returns.csv      # Log returns
├── 08_daily_funding_rate.csv     # Daily funding rates
├── 09_daily_funding_annualized.csv
├── 10_market_indicators.csv      # Market indicators
├── 11_analysis_dataset.csv       # Final analysis dataset
├── BACBB_Analysis.py             # Main analysis code
├── bacbb_returns.csv             # BACBB strategy returns
├── bacb_returns.csv              # BACB strategy returns
└── sample_*.png                  # Visualization outputs
```

---

## Academic Contributions

1. **First Application of Bad Beta to Crypto:** 암호화폐 시장에 Bad Beta 개념을 최초 적용

2. **VAR-based CF Beta Estimation:** VAR 모델 기반 Cash-Flow Beta 추정 방법론 제시

3. **Practical Implementation:** 실제 거래비용(0.04%) 및 펀딩비를 반영한 실무적 전략 구현

4. **Statistical Significance:** t-stat 2.79 (p=0.0054)로 1% 유의수준에서 통계적으로 유의한 초과수익 달성

---

## References

- Black, F. (1972). Capital market equilibrium with restricted borrowing. *Journal of Business*, 45(3), 444-455.

- Campbell, J. Y., & Shiller, R. J. (1988). The dividend-price ratio and expectations of future dividends and discount factors. *Review of Financial Studies*, 1(3), 195-228.

- Campbell, J. Y., & Vuolteenaho, T. (2004). Bad beta, good beta. *American Economic Review*, 94(5), 1249-1275.

- Frazzini, A., & Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1-25.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Keywords:** Cryptocurrency, Factor Investing, Bad Beta, Cash-Flow Beta, VAR Model, Campbell-Shiller Decomposition, Low Beta Anomaly

</div>
