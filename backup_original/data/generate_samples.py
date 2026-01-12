# -*- coding: utf-8 -*-
"""
BACBB vs BACB 비교 그래프 생성 (sample_ 파일명)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("Sample 그래프 생성")
print("="*60)

# 데이터 로드
bacbb_returns = pd.read_csv('bacbb_returns.csv', index_col=0, parse_dates=True)['BACBB']
bacb_returns = pd.read_csv('bacb_returns.csv', index_col=0, parse_dates=True)['BACB']

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

print(f"BACBB: Sharpe={m_bacbb['sharpe']:.2f}, t={m_bacbb['t_stat']:.2f}")
print(f"BACB: Sharpe={m_bacb['sharpe']:.2f}, t={m_bacb['t_stat']:.2f}")

# Sample 1: 누적 수익률 비교
fig, ax = plt.subplots(figsize=(14, 7))
cum_bacbb = (1 + bacbb_returns).cumprod()
cum_bacb = (1 + bacb_returns).cumprod()

ax.plot(cum_bacbb, label=f'BACBB (Sharpe: {m_bacbb["sharpe"]:.2f})', linewidth=2.5, color='blue')
ax.plot(cum_bacb, label=f'BACB (Sharpe: {m_bacb["sharpe"]:.2f})', linewidth=2, color='red', linestyle='--')
ax.axhline(1, color='black', linestyle='--', alpha=0.3)
ax.fill_between(cum_bacbb.index, 1, cum_bacbb, where=cum_bacbb > 1, alpha=0.2, color='blue')

ax.set_title('BACBB vs BACB 누적 수익률 비교', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.legend(loc='upper left', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sample_1_Cumulative_Returns.png', dpi=150)
plt.close()
print("  sample_1 저장: 누적 수익률")

# Sample 2: 연도별 수익률 비교
fig, ax = plt.subplots(figsize=(12, 6))
yearly_bacbb = bacbb_returns.groupby(bacbb_returns.index.year).apply(lambda x: (1+x).prod()-1)
yearly_bacb = bacb_returns.groupby(bacb_returns.index.year).apply(lambda x: (1+x).prod()-1)

years = sorted(set(yearly_bacbb.index) & set(yearly_bacb.index))
x = np.arange(len(years))
width = 0.35

bars1 = ax.bar(x - width/2, [yearly_bacbb.get(y, 0)*100 for y in years], width, label='BACBB', color='blue', alpha=0.8)
bars2 = ax.bar(x + width/2, [yearly_bacb.get(y, 0)*100 for y in years], width, label='BACB', color='red', alpha=0.8)

ax.axhline(0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Year')
ax.set_ylabel('Annual Return (%)')
ax.set_title('연도별 BACBB vs BACB 수익률 비교', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('sample_2_Yearly_Returns.png', dpi=150)
plt.close()
print("  sample_2 저장: 연도별 수익률")

# Sample 3: 롤링 샤프비율 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
rolling_sharpe_bacbb = bacbb_returns.rolling(252).mean() / bacbb_returns.rolling(252).std() * np.sqrt(252)
rolling_sharpe_bacb = bacb_returns.rolling(252).mean() / bacb_returns.rolling(252).std() * np.sqrt(252)

ax1.plot(rolling_sharpe_bacbb, label='BACBB', linewidth=1.5, color='blue')
ax1.plot(rolling_sharpe_bacb, label='BACB', linewidth=1.5, color='red')
ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
ax1.axhline(1, color='green', linestyle='--', alpha=0.5, label='Sharpe=1')
ax1.set_title('롤링 샤프비율 (252일)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Sharpe Ratio')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
strategies = ['BACBB', 'BACB']
sharpes = [m_bacbb['sharpe'], m_bacb['sharpe']]
sortinos = [m_bacbb['sortino'], m_bacb['sortino']]
calmars = [m_bacbb['calmar'], m_bacb['calmar']]

x = np.arange(len(strategies))
width = 0.25

bars1 = ax2.bar(x - width, sharpes, width, label='Sharpe', color='blue', alpha=0.8)
bars2 = ax2.bar(x, sortinos, width, label='Sortino', color='green', alpha=0.8)
bars3 = ax2.bar(x + width, calmars, width, label='Calmar', color='orange', alpha=0.8)

ax2.axhline(1, color='red', linestyle='--', alpha=0.5)
ax2.set_ylabel('Ratio')
ax2.set_title('위험조정 성과 지표 비교', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(strategies)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('sample_3_Sharpe_Comparison.png', dpi=150)
plt.close()
print("  sample_3 저장: 샤프비율 비교")

# Sample 4: BACBB-BACB 수익률 차이 및 상관계수
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

common_idx = bacbb_returns.index.intersection(bacb_returns.index)
bacbb_aligned = bacbb_returns.loc[common_idx]
bacb_aligned = bacb_returns.loc[common_idx]
diff_returns = bacbb_aligned - bacb_aligned

ax1 = axes[0, 0]
ax1.plot(diff_returns.rolling(21).mean(), color='purple', linewidth=1.5)
ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
ax1.fill_between(diff_returns.index, 0, diff_returns.rolling(21).mean(), 
                  where=diff_returns.rolling(21).mean() > 0, alpha=0.3, color='blue', label='BACBB 우위')
ax1.fill_between(diff_returns.index, 0, diff_returns.rolling(21).mean(), 
                  where=diff_returns.rolling(21).mean() < 0, alpha=0.3, color='red', label='BACB 우위')
ax1.set_title('BACBB - BACB 수익률 차이 (21일 이동평균)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Return Difference')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
rolling_corr = bacbb_aligned.rolling(63).corr(bacb_aligned)
ax2.plot(rolling_corr, color='green', linewidth=1.5)
ax2.axhline(rolling_corr.mean(), color='red', linestyle='--', label=f'평균: {rolling_corr.mean():.3f}')
ax2.set_title('BACBB-BACB 롤링 상관계수 (63일)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Correlation')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.hist(diff_returns.dropna(), bins=50, alpha=0.7, color='purple', edgecolor='black')
ax3.axvline(0, color='black', linestyle='--', alpha=0.5)
ax3.axvline(diff_returns.mean(), color='red', linestyle='-', linewidth=2, label=f'평균: {diff_returns.mean()*100:.3f}%')
ax3.set_title('BACBB - BACB 일간 수익률 차이 분포', fontsize=11, fontweight='bold')
ax3.set_xlabel('Return Difference')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.scatter(bacb_aligned, bacbb_aligned, alpha=0.3, s=10)
ax4.plot([-0.1, 0.1], [-0.1, 0.1], 'r--', label='45 line')
ax4.set_xlabel('BACB Daily Return')
ax4.set_ylabel('BACBB Daily Return')
ax4.set_title(f'BACB vs BACBB 일간 수익률 (상관: {bacbb_aligned.corr(bacb_aligned):.3f})', fontsize=11, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sample_4_BAB_BACBB_Difference.png', dpi=150)
plt.close()
print("  sample_4 저장: BACB-BACBB 차이 분석")

# Sample 5: 회귀분석
fig, ax = plt.subplots(figsize=(10, 8))

slope_reg, intercept_reg, r_reg, p_reg, _ = stats.linregress(bacb_aligned, bacbb_aligned)
ax.scatter(bacb_aligned, bacbb_aligned, alpha=0.3, s=15)
x_reg = np.linspace(bacb_aligned.min(), bacb_aligned.max(), 100)
ax.plot(x_reg, slope_reg * x_reg + intercept_reg, 'r-', linewidth=2, 
         label=f'y = {slope_reg:.3f}x + {intercept_reg:.5f}\nR2 = {r_reg**2:.3f}')
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.axvline(0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('BACB Return', fontsize=12)
ax.set_ylabel('BACBB Return', fontsize=12)
ax.set_title('BACBB on BACB 회귀분석', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sample_5_Regression.png', dpi=150)
plt.close()
print("  sample_5 저장: 회귀분석")

# Sample 6: Drawdown 비교
fig, ax = plt.subplots(figsize=(14, 6))

cum_bacbb_dd = (1 + bacbb_aligned).cumprod()
cum_bacb_dd = (1 + bacb_aligned).cumprod()
dd_bacbb = (cum_bacbb_dd - cum_bacbb_dd.cummax()) / cum_bacbb_dd.cummax()
dd_bacb = (cum_bacb_dd - cum_bacb_dd.cummax()) / cum_bacb_dd.cummax()

ax.fill_between(dd_bacbb.index, dd_bacbb*100, 0, alpha=0.5, color='blue', label=f'BACBB (MDD: {dd_bacbb.min()*100:.1f}%)')
ax.fill_between(dd_bacb.index, dd_bacb*100, 0, alpha=0.5, color='red', label=f'BACB (MDD: {dd_bacb.min()*100:.1f}%)')
ax.set_title('Drawdown 비교', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sample_6_Drawdown.png', dpi=150)
plt.close()
print("  sample_6 저장: Drawdown 비교")

# Sample 7: 성과 요약 테이블
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

summary_text = f"""
BACBB vs BACB 성과 비교 요약
============================

                    BACBB       BACB         차이
연수익률:          {m_bacbb['ann_ret']*100:6.2f}%    {m_bacb['ann_ret']*100:6.2f}%    {(m_bacbb['ann_ret']-m_bacb['ann_ret'])*100:+6.2f}%
변동성:            {m_bacbb['ann_vol']*100:6.2f}%    {m_bacb['ann_vol']*100:6.2f}%    {(m_bacbb['ann_vol']-m_bacb['ann_vol'])*100:+6.2f}%
샤프비율:          {m_bacbb['sharpe']:6.2f}      {m_bacb['sharpe']:6.2f}      {m_bacbb['sharpe']-m_bacb['sharpe']:+6.2f}
소르티노:          {m_bacbb['sortino']:6.2f}      {m_bacb['sortino']:6.2f}      {m_bacbb['sortino']-m_bacb['sortino']:+6.2f}
칼마:              {m_bacbb['calmar']:6.2f}      {m_bacb['calmar']:6.2f}      {m_bacbb['calmar']-m_bacb['calmar']:+6.2f}
MDD:               {m_bacbb['mdd']*100:6.2f}%    {m_bacb['mdd']*100:6.2f}%    {(m_bacbb['mdd']-m_bacb['mdd'])*100:+6.2f}%
승률:              {m_bacbb['win_rate']*100:6.1f}%    {m_bacb['win_rate']*100:6.1f}%    {(m_bacbb['win_rate']-m_bacb['win_rate'])*100:+6.1f}%
t-stat:            {m_bacbb['t_stat']:6.2f}      {m_bacb['t_stat']:6.2f}
p-value:           {m_bacbb['p_value']:6.4f}    {m_bacb['p_value']:6.4f}

상관계수:          {bacbb_aligned.corr(bacb_aligned):6.3f}

결론: BACBB가 BACB 대비 샤프비율 {m_bacbb['sharpe']-m_bacb['sharpe']:.2f} 우위
      통계적 유의성: {'***' if m_bacbb['p_value'] < 0.01 else '**' if m_bacbb['p_value'] < 0.05 else '*' if m_bacbb['p_value'] < 0.1 else 'N/S'}
"""
ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('sample_7_Summary.png', dpi=150)
plt.close()
print("  sample_7 저장: 성과 요약")

# Sample 8: 수익률 분포 비교
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(bacbb_returns*100, bins=50, alpha=0.6, color='blue', edgecolor='black', label='BACBB', density=True)
ax.hist(bacb_returns*100, bins=50, alpha=0.6, color='red', edgecolor='black', label='BACB', density=True)
ax.axvline(x=bacbb_returns.mean()*100, color='blue', linestyle='--', linewidth=2, label=f'BACBB Mean: {bacbb_returns.mean()*100:.3f}%')
ax.axvline(x=bacb_returns.mean()*100, color='red', linestyle='--', linewidth=2, label=f'BACB Mean: {bacb_returns.mean()*100:.3f}%')
ax.set_title('BACBB vs BACB 일간 수익률 분포', fontsize=14, fontweight='bold')
ax.set_xlabel('Daily Return (%)')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig('sample_8_Distribution.png', dpi=150)
plt.close()
print("  sample_8 저장: 수익률 분포")

print("\n모든 Sample 그래프 생성 완료! (sample_1 ~ sample_8)")
