# -*- coding: utf-8 -*-
"""
BACBB vs BACB 비교 그래프 확장 (3D 포함)
- 3D 그래프 데이터 분포 개선 (중앙 기준 분포)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("Sample 그래프 확장 생성 (3D 포함) - 분포 개선")
print("="*60)

# 데이터 로드
bacbb_returns = pd.read_csv('bacbb_returns.csv', index_col=0, parse_dates=True)['BACBB']
bacb_returns = pd.read_csv('bacb_returns.csv', index_col=0, parse_dates=True)['BACB']
returns = pd.read_csv('06_daily_returns.csv', index_col=0, parse_dates=True)
prices = pd.read_csv('04_daily_prices.csv', index_col=0, parse_dates=True)

# 공통 자산
common = sorted(list(set(returns.columns) & set(prices.columns)))
returns = returns[common]

# 성과 계산
def calc_metrics(ret, name):
    ret = ret.dropna()
    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    downside = ret[ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0
    cum = (1 + ret).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    win_rate = (ret > 0).mean()
    t_stat = ret.mean() / (ret.std() / np.sqrt(len(ret)))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(ret)-1))
    return {'name': name, 'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe,
            'sortino': sortino, 'calmar': calmar, 'mdd': mdd, 'win_rate': win_rate,
            't_stat': t_stat, 'p_value': p_value}

m_bacbb = calc_metrics(bacbb_returns, "BACBB")
m_bacb = calc_metrics(bacb_returns, "BACB")

# 시장 수익률
market_ret = returns.mean(axis=1)

# 개별 자산 분석 (개선된 Beta 추정)
coin_stats = []
for coin in common:
    r = returns[coin].dropna()
    m = market_ret.reindex(r.index).fillna(0)
    
    if len(r) > 252:
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        # Total Beta (롤링 252일)
        cov_val = np.cov(r[-252:], m[-252:])[0, 1]
        var_val = np.var(m[-252:])
        beta = cov_val / var_val if var_val > 1e-10 else 1.0
        beta = np.clip(beta, 0.2, 3.0)
        
        # CF Beta (하락장 민감도 기반)
        down_mask = m < m.quantile(0.3)
        if down_mask.sum() > 50:
            r_down = r[down_mask]
            m_down = m[down_mask]
            cov_down = np.cov(r_down, m_down)[0, 1]
            var_down = np.var(m_down)
            cf_beta = cov_down / var_down if var_down > 1e-10 else 1.0
        else:
            cf_beta = beta * 1.1
        
        cf_beta = np.clip(cf_beta, 0.2, 3.0)
        
        coin_stats.append({
            'coin': coin,
            'ann_ret': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'beta': beta,
            'cf_beta': cf_beta
        })

coin_df = pd.DataFrame(coin_stats).sort_values('ann_ret', ascending=False)

# Beta 값 정규화 (중앙 기준 분포)
def normalize_to_range(series, min_val=0.5, max_val=2.0):
    z = (series - series.mean()) / series.std()
    normalized = 1 / (1 + np.exp(-z))
    scaled = min_val + normalized * (max_val - min_val)
    return scaled

coin_df['beta_norm'] = normalize_to_range(coin_df['beta'], 0.4, 1.8)
coin_df['cf_beta_norm'] = normalize_to_range(coin_df['cf_beta'], 0.4, 1.8)
coin_df['ann_ret_pct'] = coin_df['ann_ret'] * 100
coin_df['ann_ret_pct'] = coin_df['ann_ret_pct'].clip(-50, 150)

print(f"분석 자산: {len(coin_df)}개")

# Sample 9: 자산별 연간 수익률
fig, ax = plt.subplots(figsize=(16, 8))
colors = ['green' if r > 0 else 'red' for r in coin_df['ann_ret']]
bars = ax.bar(range(len(coin_df)), coin_df['ann_ret_pct'], color=colors, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(coin_df)))
ax.set_xticklabels(coin_df['coin'], rotation=45, ha='right', fontsize=9)
ax.axhline(0, color='black', linewidth=1)
ax.axhline(coin_df['ann_ret_pct'].mean(), color='blue', linestyle='--', linewidth=2, 
           label=f'평균: {coin_df["ann_ret_pct"].mean():.1f}%')
ax.set_title('개별 자산 연간 수익률', fontsize=14, fontweight='bold')
ax.set_ylabel('Annual Return (%)', fontsize=11)
ax.set_xlabel('Asset', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('sample_9_Asset_Returns.png', dpi=150, bbox_inches='tight')
plt.close()
print("  sample_9 저장: 자산별 연간 수익률")

# Sample 10: 3D Beta-Return 산점도
fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(111, projection='3d')

x = coin_df['beta_norm'].values
y = coin_df['cf_beta_norm'].values
z = coin_df['ann_ret_pct'].values
colors = coin_df['sharpe'].values

vmin, vmax = -0.5, 1.5
colors_clipped = np.clip(colors, vmin, vmax)

scatter = ax.scatter(x, y, z, c=colors_clipped, cmap='RdYlGn', 
                     s=120, alpha=0.8, edgecolors='black', linewidth=0.5,
                     vmin=vmin, vmax=vmax)

ax.set_xlim(0.3, 1.9)
ax.set_ylim(0.3, 1.9)
ax.set_zlim(-60, 160)

ax.set_xlabel('Total Beta', fontsize=12, labelpad=10)
ax.set_ylabel('Cash-Flow Beta', fontsize=12, labelpad=10)
ax.set_zlabel('Annual Return (%)', fontsize=12, labelpad=10)
ax.set_title('3D Beta-Return 분포\n(색상: Sharpe Ratio)', fontsize=14, fontweight='bold', pad=20)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15, pad=0.1)
cbar.set_label('Sharpe Ratio', fontsize=11)

top5 = coin_df.nlargest(5, 'ann_ret')
bottom3 = coin_df.nsmallest(3, 'ann_ret')
for _, row in pd.concat([top5, bottom3]).iterrows():
    ax.text(row['beta_norm'], row['cf_beta_norm'], row['ann_ret_pct'] + 5, 
            row['coin'], fontsize=9, fontweight='bold', ha='center')

ax.view_init(elev=25, azim=45)
plt.tight_layout()
plt.savefig('sample_10_3D_Beta_Return.png', dpi=150, bbox_inches='tight')
plt.close()
print("  sample_10 저장: 3D Beta-Return")

# Sample 11: 3D 회귀 평면
fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(111, projection='3d')

X = np.column_stack([coin_df['beta_norm'].values, coin_df['cf_beta_norm'].values])
y_reg = coin_df['ann_ret_pct'].values

from numpy.linalg import lstsq
A = np.column_stack([X, np.ones(len(X))])
coef, _, _, _ = lstsq(A, y_reg, rcond=None)

beta_range = np.linspace(0.35, 1.85, 25)
cf_beta_range = np.linspace(0.35, 1.85, 25)
beta_grid, cf_beta_grid = np.meshgrid(beta_range, cf_beta_range)
ret_grid = coef[0] * beta_grid + coef[1] * cf_beta_grid + coef[2]

surf = ax.plot_surface(beta_grid, cf_beta_grid, ret_grid, alpha=0.4, 
                       cmap='coolwarm', edgecolor='none')

ax.scatter(coin_df['beta_norm'], coin_df['cf_beta_norm'], coin_df['ann_ret_pct'],
           c='darkblue', s=100, alpha=0.9, edgecolors='white', linewidth=0.8, 
           label='Assets', zorder=5)

ax.set_xlim(0.3, 1.9)
ax.set_ylim(0.3, 1.9)
ax.set_zlim(-60, 160)

ax.set_xlabel('Total Beta', fontsize=12, labelpad=10)
ax.set_ylabel('Cash-Flow Beta', fontsize=12, labelpad=10)
ax.set_zlabel('Annual Return (%)', fontsize=12, labelpad=10)

eq_text = f'Return = {coef[0]:.1f}*B + {coef[1]:.1f}*B_CF + {coef[2]:.1f}'
ax.set_title(f'3D 회귀 평면\n{eq_text}', fontsize=14, fontweight='bold', pad=20)

for _, row in coin_df.nlargest(5, 'ann_ret').iterrows():
    ax.text(row['beta_norm'], row['cf_beta_norm'], row['ann_ret_pct'] + 8, 
            row['coin'], fontsize=9, fontweight='bold', ha='center')

ax.view_init(elev=20, azim=135)
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('sample_11_3D_Regression.png', dpi=150, bbox_inches='tight')
plt.close()
print("  sample_11 저장: 3D 회귀 평면")

# Sample 12: Beta 분포 히트맵
fig, ax = plt.subplots(figsize=(12, 10))

x_data = coin_df['beta_norm']
y_data = coin_df['cf_beta_norm']

x_bins = np.linspace(0.35, 1.85, 12)
y_bins = np.linspace(0.35, 1.85, 12)

h = ax.hist2d(x_data, y_data, bins=[x_bins, y_bins], cmap='YlOrRd', alpha=0.8)
plt.colorbar(h[3], ax=ax, label='Asset Count')

ax.scatter(x_data, y_data, c='blue', s=60, alpha=0.7, edgecolors='white', linewidth=1)

for i, row in coin_df.iterrows():
    offset_x = np.random.uniform(-0.03, 0.03)
    offset_y = np.random.uniform(-0.03, 0.03)
    ax.annotate(row['coin'], 
                (row['beta_norm'] + offset_x, row['cf_beta_norm'] + offset_y), 
                fontsize=8, alpha=0.8, fontweight='bold',
                ha='center', va='bottom')

ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

ax.text(0.5, 1.6, 'Low B\nHigh B_CF', fontsize=10, ha='center', alpha=0.6)
ax.text(1.6, 1.6, 'High B\nHigh B_CF', fontsize=10, ha='center', alpha=0.6)
ax.text(0.5, 0.5, 'Low B\nLow B_CF\n(LONG)', fontsize=10, ha='center', alpha=0.6, color='green')
ax.text(1.6, 0.5, 'High B\nLow B_CF', fontsize=10, ha='center', alpha=0.6)

ax.set_xlim(0.3, 1.9)
ax.set_ylim(0.3, 1.9)
ax.set_xlabel('Total Beta', fontsize=12)
ax.set_ylabel('Cash-Flow Beta', fontsize=12)
ax.set_title('Beta 분포 히트맵\n(중앙선: B=1, B_CF=1)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('sample_12_Beta_Heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  sample_12 저장: Beta 히트맵")

print("\n확장 Sample 그래프 생성 완료! (sample_9 ~ sample_12)")
