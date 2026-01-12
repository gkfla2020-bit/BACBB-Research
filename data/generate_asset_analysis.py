"""
BACBB 자산별 분석 - 고베타/저베타 코인 및 성과 표 생성
기존 코드(샤프 1.04) 기반
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("BACBB 자산별 분석")
print("="*70)

# 데이터 로드
prices = pd.read_csv('04_daily_prices.csv', index_col=0, parse_dates=True)
returns = pd.read_csv('06_daily_returns.csv', index_col=0, parse_dates=True)
volumes = pd.read_csv('05_daily_volumes.csv', index_col=0, parse_dates=True)

common = sorted(list(set(prices.columns) & set(returns.columns) & set(volumes.columns)))
returns = returns[common].fillna(0).clip(-0.5, 0.5)

# 기존 분석 결과 로드
bacbb_ret = pd.read_csv('bacbb_returns.csv', index_col=0, parse_dates=True)
bacb_ret = pd.read_csv('bacb_returns.csv', index_col=0, parse_dates=True)

print(f"분석 자산: {len(common)}개")
print(f"분석 기간: {returns.index[0].strftime('%Y-%m-%d')} ~ {returns.index[-1].strftime('%Y-%m-%d')}")

# =============================================================================
# CF Beta 재계산 (기존 코드와 동일)
# =============================================================================
from numpy.linalg import inv

# 시장 수익률
vol_weights = volumes[common].div(volumes[common].sum(axis=1), axis=0).fillna(1/len(common))
market_ret = (returns * vol_weights).sum(axis=1)

# 상태변수
market_excess = market_ret - 0.0001
valuation = -prices[common].mean(axis=1).pct_change(periods=500).fillna(0).clip(-1, 1)
term_spread = pd.Series(0.01, index=returns.index)

state_vars = pd.DataFrame({
    'market_excess': market_excess.clip(-0.2, 0.2),
    'term_spread': term_spread / 100,
    'valuation': valuation
}, index=returns.index).fillna(0)

# VAR 모델
print("\n[VAR 모델 추정 중...]")
cf_news = pd.Series(index=returns.index, dtype=float)
rho = 0.997
window = 252

for i in range(window, len(returns.index)):
    z = state_vars.iloc[i-window:i].values
    z_lag = z[:-1]
    z_curr = z[1:]
    X = np.column_stack([np.ones(len(z_lag)), z_lag])
    
    try:
        beta = np.linalg.lstsq(X, z_curr, rcond=None)[0]
        A = beta[1:].T
        residuals = z_curr - X @ beta
        u_t = residuals[-1]
        
        I = np.eye(3)
        inv_term = inv(I - rho * A)
        e1 = np.array([1, 0, 0])
        
        dr = e1 @ (rho * A @ inv_term) @ u_t
        cf_news.iloc[i] = u_t[0] + dr
    except:
        continue

cf_news = cf_news.fillna(method='ffill').fillna(0)

# CF Beta 추정
print("[CF Beta 추정 중...]")
cf_beta = pd.DataFrame(index=returns.index, columns=common, dtype=float)
beta_window = 60

for i in range(beta_window, len(returns.index), 5):
    cf_window = cf_news.iloc[i-beta_window:i].values
    var_cf = np.var(cf_window)
    
    if var_cf < 1e-12:
        continue
    
    for j, col in enumerate(common):
        r_window = returns[col].iloc[i-beta_window:i].values
        if np.std(r_window) > 0:
            cov_cf = np.cov(r_window, cf_window)[0, 1]
            cf_beta.iloc[i, j] = cov_cf / var_cf

# bfill 포함 (기존 코드와 동일)
cf_beta = cf_beta.ffill().bfill()
cf_beta = cf_beta.astype(float) * 0.6 + 0.4
cf_beta = cf_beta.clip(0.1, 3.0)

# FP Beta 추정
print("[FP Beta 추정 중...]")
fp_beta = pd.DataFrame(index=returns.index, columns=common, dtype=float)
vol_window = 252
corr_window = 5

for i in range(vol_window, len(returns.index)):
    mkt_vol = market_ret.iloc[i-vol_window:i].std()
    if mkt_vol < 1e-10:
        continue
    
    m_corr = market_ret.iloc[i-corr_window:i].values
    
    for j, col in enumerate(common):
        r_vol = returns[col].iloc[i-vol_window:i].std()
        if r_vol < 1e-10:
            continue
        
        r_corr = returns[col].iloc[i-corr_window:i].values
        if np.std(r_corr) > 0 and np.std(m_corr) > 0:
            rho_5 = np.corrcoef(r_corr, m_corr)[0, 1]
        else:
            rho_5 = 0.5
        
        fp_beta.iloc[i, j] = rho_5 * (r_vol / mkt_vol)

fp_beta = fp_beta.ffill().bfill()
fp_beta = fp_beta.astype(float) * 0.6 + 0.4
fp_beta = fp_beta.clip(0.1, 3.0)

print("베타 추정 완료!")

# =============================================================================
# 자산별 분석
# =============================================================================
print("\n" + "="*70)
print("자산별 분석")
print("="*70)

asset_analysis = []

for coin in common:
    r = returns[coin].dropna()
    if len(r) < 252:
        continue
    
    # 성과 지표
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    cum_ret = (1 + r).cumprod()
    total_ret = cum_ret.iloc[-1] - 1
    
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    mdd = drawdown.min()
    
    # 베타
    avg_cf_beta = cf_beta[coin].dropna().mean()
    avg_fp_beta = fp_beta[coin].dropna().mean()
    
    # 시장 상관관계
    common_idx = r.index.intersection(market_ret.index)
    mkt_corr = r.loc[common_idx].corr(market_ret.loc[common_idx])
    
    asset_analysis.append({
        'Coin': coin,
        'CF_Beta': avg_cf_beta,
        'FP_Beta': avg_fp_beta,
        'Ann_Return': ann_ret,
        'Volatility': ann_vol,
        'Sharpe': sharpe,
        'Total_Return': total_ret,
        'MDD': mdd,
        'Mkt_Corr': mkt_corr
    })

asset_df = pd.DataFrame(asset_analysis)
asset_df = asset_df.sort_values('CF_Beta')

# =============================================================================
# 저베타/고베타 그룹 분류
# =============================================================================
n = len(asset_df)
n_group = n // 3

low_beta = asset_df.head(n_group).copy()
mid_beta = asset_df.iloc[n_group:2*n_group].copy()
high_beta = asset_df.tail(n_group).copy()

low_beta['Group'] = 'Low CF Beta (Long)'
mid_beta['Group'] = 'Mid CF Beta'
high_beta['Group'] = 'High CF Beta (Short)'

print("\n[저 CF Beta 코인 - Long 후보]")
print("-" * 70)
for _, row in low_beta.iterrows():
    print(f"  {row['Coin']:6s}: CF Beta {row['CF_Beta']:.3f}, 연 {row['Ann_Return']*100:>6.1f}%, 샤프 {row['Sharpe']:.2f}, MDD {row['MDD']*100:.1f}%")

print("\n[고 CF Beta 코인 - Short 후보]")
print("-" * 70)
for _, row in high_beta.iterrows():
    print(f"  {row['Coin']:6s}: CF Beta {row['CF_Beta']:.3f}, 연 {row['Ann_Return']*100:>6.1f}%, 샤프 {row['Sharpe']:.2f}, MDD {row['MDD']*100:.1f}%")

# =============================================================================
# 그룹별 성과 요약
# =============================================================================
print("\n" + "="*70)
print("그룹별 성과 요약")
print("="*70)

group_summary = []
for name, group in [('Low CF Beta', low_beta), ('Mid CF Beta', mid_beta), ('High CF Beta', high_beta)]:
    group_summary.append({
        'Group': name,
        'N_Coins': len(group),
        'Avg_CF_Beta': group['CF_Beta'].mean(),
        'Avg_Return': group['Ann_Return'].mean(),
        'Avg_Sharpe': group['Sharpe'].mean(),
        'Avg_Vol': group['Volatility'].mean(),
        'Avg_MDD': group['MDD'].mean()
    })

group_df = pd.DataFrame(group_summary)
print(group_df.to_string(index=False))

# =============================================================================
# 표 저장
# =============================================================================
print("\n" + "="*70)
print("표 저장")
print("="*70)

# Table: 전체 자산 분석
asset_df_save = asset_df.copy()
asset_df_save['Ann_Return'] = asset_df_save['Ann_Return'].apply(lambda x: f"{x*100:.2f}%")
asset_df_save['Volatility'] = asset_df_save['Volatility'].apply(lambda x: f"{x*100:.2f}%")
asset_df_save['Sharpe'] = asset_df_save['Sharpe'].apply(lambda x: f"{x:.2f}")
asset_df_save['Total_Return'] = asset_df_save['Total_Return'].apply(lambda x: f"{x*100:.1f}%")
asset_df_save['MDD'] = asset_df_save['MDD'].apply(lambda x: f"{x*100:.1f}%")
asset_df_save['CF_Beta'] = asset_df_save['CF_Beta'].apply(lambda x: f"{x:.3f}")
asset_df_save['FP_Beta'] = asset_df_save['FP_Beta'].apply(lambda x: f"{x:.3f}")
asset_df_save['Mkt_Corr'] = asset_df_save['Mkt_Corr'].apply(lambda x: f"{x:.3f}")

asset_df_save.to_csv('Table_Asset_Analysis.csv', index=False)
print("  Table_Asset_Analysis.csv 저장 완료")

# Table: 저베타 코인
low_beta_save = low_beta[['Coin', 'CF_Beta', 'FP_Beta', 'Ann_Return', 'Volatility', 'Sharpe', 'MDD']].copy()
low_beta_save.to_csv('Table_Low_Beta_Coins.csv', index=False)
print("  Table_Low_Beta_Coins.csv 저장 완료")

# Table: 고베타 코인
high_beta_save = high_beta[['Coin', 'CF_Beta', 'FP_Beta', 'Ann_Return', 'Volatility', 'Sharpe', 'MDD']].copy()
high_beta_save.to_csv('Table_High_Beta_Coins.csv', index=False)
print("  Table_High_Beta_Coins.csv 저장 완료")

# Table: 그룹별 요약
group_df.to_csv('Table_Group_Summary.csv', index=False)
print("  Table_Group_Summary.csv 저장 완료")

# =============================================================================
# 시각화
# =============================================================================
print("\n" + "="*70)
print("시각화")
print("="*70)

# Figure: CF Beta vs Return
fig, ax = plt.subplots(figsize=(12, 8))

colors = {'Low CF Beta (Long)': 'green', 'Mid CF Beta': 'gray', 'High CF Beta (Short)': 'red'}
all_data = pd.concat([low_beta, mid_beta, high_beta])

for group_name in colors.keys():
    group_data = all_data[all_data['Group'] == group_name]
    ax.scatter(group_data['CF_Beta'], group_data['Ann_Return'] * 100, 
               s=group_data['Volatility'] * 150, alpha=0.6, 
               c=colors[group_name], label=group_name, edgecolor='black')

for _, row in all_data.iterrows():
    ax.annotate(row['Coin'], (row['CF_Beta'], row['Ann_Return'] * 100), 
                fontsize=8, alpha=0.8, ha='center', va='bottom')

ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('CF Beta (Bad Beta)', fontsize=12)
ax.set_ylabel('Annual Return (%)', fontsize=12)
ax.set_title('CF Beta vs 연수익률 (버블 크기 = 변동성)', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Figure_Asset_CF_Beta.png', dpi=150)
plt.close()
print("  Figure_Asset_CF_Beta.png 저장 완료")

# Figure: 그룹별 성과 비교
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# 연수익률
ax = axes[0]
groups = ['Low CF Beta', 'Mid CF Beta', 'High CF Beta']
returns_by_group = [low_beta['Ann_Return'].mean() * 100, 
                    mid_beta['Ann_Return'].mean() * 100, 
                    high_beta['Ann_Return'].mean() * 100]
colors_bar = ['green', 'gray', 'red']
ax.bar(groups, returns_by_group, color=colors_bar, alpha=0.7, edgecolor='black')
ax.set_ylabel('Annual Return (%)')
ax.set_title('그룹별 평균 연수익률')
ax.axhline(0, color='black', linestyle='-', alpha=0.3)
ax.grid(True, alpha=0.3, axis='y')

# 샤프비율
ax = axes[1]
sharpes_by_group = [low_beta['Sharpe'].mean(), 
                    mid_beta['Sharpe'].mean(), 
                    high_beta['Sharpe'].mean()]
ax.bar(groups, sharpes_by_group, color=colors_bar, alpha=0.7, edgecolor='black')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('그룹별 평균 샤프비율')
ax.axhline(0, color='black', linestyle='-', alpha=0.3)
ax.grid(True, alpha=0.3, axis='y')

# MDD
ax = axes[2]
mdds_by_group = [low_beta['MDD'].mean() * 100, 
                 mid_beta['MDD'].mean() * 100, 
                 high_beta['MDD'].mean() * 100]
ax.bar(groups, mdds_by_group, color=colors_bar, alpha=0.7, edgecolor='black')
ax.set_ylabel('MDD (%)')
ax.set_title('그룹별 평균 MDD')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Figure_Group_Performance.png', dpi=150)
plt.close()
print("  Figure_Group_Performance.png 저장 완료")

# Figure: 베타 분포
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(asset_df['CF_Beta'].astype(float), bins=20, alpha=0.7, color='blue', edgecolor='black')
ax.axvline(asset_df['CF_Beta'].astype(float).mean(), color='red', linestyle='--', 
           label=f'Mean: {asset_df["CF_Beta"].astype(float).mean():.3f}')
ax.set_xlabel('CF Beta')
ax.set_ylabel('Frequency')
ax.set_title('CF Beta 분포')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(asset_df['FP_Beta'].astype(float), bins=20, alpha=0.7, color='orange', edgecolor='black')
ax.axvline(asset_df['FP_Beta'].astype(float).mean(), color='red', linestyle='--', 
           label=f'Mean: {asset_df["FP_Beta"].astype(float).mean():.3f}')
ax.set_xlabel('FP Beta')
ax.set_ylabel('Frequency')
ax.set_title('FP Beta 분포')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Figure_Beta_Distribution.png', dpi=150)
plt.close()
print("  Figure_Beta_Distribution.png 저장 완료")

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
