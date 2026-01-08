# -*- coding: utf-8 -*-
"""
BACBB Out-of-Sample 검증 그래프 생성
====================================
In-Sample vs Out-of-Sample 성과 비교
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
print("Out-of-Sample 검증 그래프 생성")
print("="*60)

# 데이터 로드
bacbb = pd.read_csv('bacbb_returns.csv', index_col=0, parse_dates=True)['BACBB']
bacb = pd.read_csv('bacb_returns.csv', index_col=0, parse_dates=True)['BACB']

# 공통 인덱스
common_idx = bacbb.index.intersection(bacb.index)
bacbb = bacbb.loc[common_idx]
bacb = bacb.loc[common_idx]

print(f"전체 기간: {bacbb.index[0].strftime('%Y-%m-%d')} ~ {bacbb.index[-1].strftime('%Y-%m-%d')}")
print(f"거래일 수: {len(bacbb)}")

# In-Sample / Out-of-Sample 분할 (50:50)
split_idx = len(bacbb) // 2
split_date = bacbb.index[split_idx]

bacbb_is = bacbb.iloc[:split_idx]
bacbb_oos = bacbb.iloc[split_idx:]
bacb_is = bacb.iloc[:split_idx]
bacb_oos = bacb.iloc[split_idx:]

print(f"\nIn-Sample: {bacbb_is.index[0].strftime('%Y-%m-%d')} ~ {bacbb_is.index[-1].strftime('%Y-%m-%d')} ({len(bacbb_is)}일)")
print(f"Out-of-Sample: {bacbb_oos.index[0].strftime('%Y-%m-%d')} ~ {bacbb_oos.index[-1].strftime('%Y-%m-%d')} ({len(bacbb_oos)}일)")

# 성과 계산 함수
def calc_metrics(ret):
    ret = ret.dropna()
    n = len(ret)
    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + ret).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    t_stat = ret.mean() / (ret.std() / np.sqrt(n)) if ret.std() > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-1))
    sortino = ann_ret / (ret[ret<0].std() * np.sqrt(252)) if len(ret[ret<0]) > 0 else 0
    win_rate = (ret > 0).mean()
    return {
        'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe,
        'sortino': sortino, 'mdd': mdd, 't_stat': t_stat, 
        'p_value': p_value, 'win_rate': win_rate
    }

# 각 기간별 성과 계산
m_bacbb_is = calc_metrics(bacbb_is)
m_bacbb_oos = calc_metrics(bacbb_oos)
m_bacb_is = calc_metrics(bacb_is)
m_bacb_oos = calc_metrics(bacb_oos)

print("\n[BACBB 성과]")
print(f"  IS:  연 {m_bacbb_is['ann_ret']*100:.2f}%, 샤프 {m_bacbb_is['sharpe']:.2f}, p={m_bacbb_is['p_value']:.4f}")
print(f"  OOS: 연 {m_bacbb_oos['ann_ret']*100:.2f}%, 샤프 {m_bacbb_oos['sharpe']:.2f}, p={m_bacbb_oos['p_value']:.4f}")

print("\n[BACB 성과]")
print(f"  IS:  연 {m_bacb_is['ann_ret']*100:.2f}%, 샤프 {m_bacb_is['sharpe']:.2f}, p={m_bacb_is['p_value']:.4f}")
print(f"  OOS: 연 {m_bacb_oos['ann_ret']*100:.2f}%, 샤프 {m_bacb_oos['sharpe']:.2f}, p={m_bacb_oos['p_value']:.4f}")

# Figure 1: In-Sample vs Out-of-Sample 누적 수익률
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1-1: BACBB 누적 수익률 (IS vs OOS)
ax1 = axes[0, 0]
cum_is = (1 + bacbb_is).cumprod()
cum_oos = (1 + bacbb_oos).cumprod()

ax1.plot(cum_is.index, cum_is.values, 'b-', linewidth=2, label=f'In-Sample (Sharpe: {m_bacbb_is["sharpe"]:.2f})')
ax1.plot(cum_oos.index, cum_oos.values, 'r-', linewidth=2, label=f'Out-of-Sample (Sharpe: {m_bacbb_oos["sharpe"]:.2f})')
ax1.axvline(split_date, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Split Date')
ax1.axhline(1, color='black', linestyle='-', alpha=0.3)
ax1.fill_between(cum_is.index, 1, cum_is.values, alpha=0.2, color='blue')
ax1.fill_between(cum_oos.index, 1, cum_oos.values, alpha=0.2, color='red')
ax1.set_title('BACBB 누적 수익률: In-Sample vs Out-of-Sample', fontsize=13, fontweight='bold')
ax1.set_ylabel('Cumulative Return')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# 1-2: BACB 누적 수익률 (IS vs OOS)
ax2 = axes[0, 1]
cum_is_b = (1 + bacb_is).cumprod()
cum_oos_b = (1 + bacb_oos).cumprod()

ax2.plot(cum_is_b.index, cum_is_b.values, 'b-', linewidth=2, label=f'In-Sample (Sharpe: {m_bacb_is["sharpe"]:.2f})')
ax2.plot(cum_oos_b.index, cum_oos_b.values, 'r-', linewidth=2, label=f'Out-of-Sample (Sharpe: {m_bacb_oos["sharpe"]:.2f})')
ax2.axvline(split_date, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Split Date')
ax2.axhline(1, color='black', linestyle='-', alpha=0.3)
ax2.fill_between(cum_is_b.index, 1, cum_is_b.values, alpha=0.2, color='blue')
ax2.fill_between(cum_oos_b.index, 1, cum_oos_b.values, alpha=0.2, color='red')
ax2.set_title('BACB 누적 수익률: In-Sample vs Out-of-Sample', fontsize=13, fontweight='bold')
ax2.set_ylabel('Cumulative Return')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

# 1-3: 성과 지표 비교 바 차트
ax3 = axes[1, 0]
metrics = ['연수익률(%)', '샤프비율', '소르티노', 'MDD(%)']
x = np.arange(len(metrics))
width = 0.2

bacbb_is_vals = [m_bacbb_is['ann_ret']*100, m_bacbb_is['sharpe'], m_bacbb_is['sortino'], abs(m_bacbb_is['mdd'])*100]
bacbb_oos_vals = [m_bacbb_oos['ann_ret']*100, m_bacbb_oos['sharpe'], m_bacbb_oos['sortino'], abs(m_bacbb_oos['mdd'])*100]
bacb_is_vals = [m_bacb_is['ann_ret']*100, m_bacb_is['sharpe'], m_bacb_is['sortino'], abs(m_bacb_is['mdd'])*100]
bacb_oos_vals = [m_bacb_oos['ann_ret']*100, m_bacb_oos['sharpe'], m_bacb_oos['sortino'], abs(m_bacb_oos['mdd'])*100]

bars1 = ax3.bar(x - 1.5*width, bacbb_is_vals, width, label='BACBB IS', color='blue', alpha=0.8)
bars2 = ax3.bar(x - 0.5*width, bacbb_oos_vals, width, label='BACBB OOS', color='lightblue', alpha=0.8, edgecolor='blue')
bars3 = ax3.bar(x + 0.5*width, bacb_is_vals, width, label='BACB IS', color='red', alpha=0.8)
bars4 = ax3.bar(x + 1.5*width, bacb_oos_vals, width, label='BACB OOS', color='lightcoral', alpha=0.8, edgecolor='red')

ax3.set_xticks(x)
ax3.set_xticklabels(metrics, fontsize=11)
ax3.set_title('성과 지표 비교: In-Sample vs Out-of-Sample', fontsize=13, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

# 1-4: t-stat 및 p-value 비교
ax4 = axes[1, 1]
categories = ['BACBB\nIn-Sample', 'BACBB\nOut-of-Sample', 'BACB\nIn-Sample', 'BACB\nOut-of-Sample']
t_stats = [m_bacbb_is['t_stat'], m_bacbb_oos['t_stat'], m_bacb_is['t_stat'], m_bacb_oos['t_stat']]
p_values = [m_bacbb_is['p_value'], m_bacbb_oos['p_value'], m_bacb_is['p_value'], m_bacb_oos['p_value']]

colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
bars = ax4.bar(categories, t_stats, color=colors, alpha=0.8, edgecolor='black')

ax4.axhline(1.96, color='red', linestyle='--', linewidth=1.5, label='5% 유의수준 (t=1.96)')
ax4.axhline(1.645, color='orange', linestyle='--', linewidth=1.5, label='10% 유의수준 (t=1.645)')

for i, (bar, p) in enumerate(zip(bars, p_values)):
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    ax4.annotate(f't={t_stats[i]:.2f}\np={p:.4f}{sig}', 
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9)

ax4.set_title('통계적 유의성 검정 (t-statistic)', fontsize=13, fontweight='bold')
ax4.set_ylabel('t-statistic')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('sample_13_OOS_Validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  sample_13_OOS_Validation.png 저장 완료")

# Figure 2: 롤링 성과 비교
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

window = 126

# 2-1: 롤링 샤프비율
ax1 = axes[0, 0]
rolling_sharpe_bacbb = (bacbb.rolling(window).mean() * 252) / (bacbb.rolling(window).std() * np.sqrt(252))
rolling_sharpe_bacb = (bacb.rolling(window).mean() * 252) / (bacb.rolling(window).std() * np.sqrt(252))

ax1.plot(rolling_sharpe_bacbb.index, rolling_sharpe_bacbb.values, 'b-', linewidth=1.5, label='BACBB', alpha=0.8)
ax1.plot(rolling_sharpe_bacb.index, rolling_sharpe_bacb.values, 'r-', linewidth=1.5, label='BACB', alpha=0.8)
ax1.axvline(split_date, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax1.axhline(1, color='green', linestyle='-', alpha=0.5, linewidth=1)
ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
ax1.axvspan(bacbb.index[0], split_date, alpha=0.1, color='blue', label='In-Sample')
ax1.axvspan(split_date, bacbb.index[-1], alpha=0.1, color='red', label='Out-of-Sample')
ax1.set_title(f'롤링 샤프비율 ({window}일)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Sharpe Ratio')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-2, 4)

# 2-2: 롤링 수익률
ax2 = axes[0, 1]
rolling_ret_bacbb = bacbb.rolling(window).mean() * 252 * 100
rolling_ret_bacb = bacb.rolling(window).mean() * 252 * 100

ax2.plot(rolling_ret_bacbb.index, rolling_ret_bacbb.values, 'b-', linewidth=1.5, label='BACBB', alpha=0.8)
ax2.plot(rolling_ret_bacb.index, rolling_ret_bacb.values, 'r-', linewidth=1.5, label='BACB', alpha=0.8)
ax2.axvline(split_date, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
ax2.axvspan(bacbb.index[0], split_date, alpha=0.1, color='blue')
ax2.axvspan(split_date, bacbb.index[-1], alpha=0.1, color='red')
ax2.set_title(f'롤링 연환산 수익률 ({window}일)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Annualized Return (%)')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

# 2-3: 롤링 변동성
ax3 = axes[1, 0]
rolling_vol_bacbb = bacbb.rolling(window).std() * np.sqrt(252) * 100
rolling_vol_bacb = bacb.rolling(window).std() * np.sqrt(252) * 100

ax3.plot(rolling_vol_bacbb.index, rolling_vol_bacbb.values, 'b-', linewidth=1.5, label='BACBB', alpha=0.8)
ax3.plot(rolling_vol_bacb.index, rolling_vol_bacb.values, 'r-', linewidth=1.5, label='BACB', alpha=0.8)
ax3.axvline(split_date, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax3.axvspan(bacbb.index[0], split_date, alpha=0.1, color='blue')
ax3.axvspan(split_date, bacbb.index[-1], alpha=0.1, color='red')
ax3.set_title(f'롤링 변동성 ({window}일)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Annualized Volatility (%)')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)

# 2-4: BACBB-BACB 수익률 차이
ax4 = axes[1, 1]
diff = bacbb - bacb
rolling_diff = diff.rolling(window).mean() * 252 * 100

ax4.plot(rolling_diff.index, rolling_diff.values, 'purple', linewidth=1.5, label='롤링 차이 (연환산)', alpha=0.8)
ax4.axvline(split_date, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax4.axhline(0, color='black', linestyle='-', alpha=0.5)
ax4.axvspan(bacbb.index[0], split_date, alpha=0.1, color='blue')
ax4.axvspan(split_date, bacbb.index[-1], alpha=0.1, color='red')

is_diff = diff.iloc[:split_idx].mean() * 252 * 100
oos_diff = diff.iloc[split_idx:].mean() * 252 * 100
ax4.axhline(is_diff, color='blue', linestyle=':', linewidth=2, alpha=0.7, label=f'IS 평균: {is_diff:.1f}%')
ax4.axhline(oos_diff, color='red', linestyle=':', linewidth=2, alpha=0.7, label=f'OOS 평균: {oos_diff:.1f}%')

ax4.set_title('BACBB - BACB 수익률 차이', fontsize=13, fontweight='bold')
ax4.set_ylabel('Return Difference (%)')
ax4.legend(loc='upper left', fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sample_14_OOS_Rolling.png', dpi=150, bbox_inches='tight')
plt.close()
print("  sample_14_OOS_Rolling.png 저장 완료")

# Figure 3: OOS 상세 분석
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3-1: 전체 기간 누적 수익률
ax1 = axes[0, 0]
cum_bacbb = (1 + bacbb).cumprod()
cum_bacb = (1 + bacb).cumprod()

ax1.plot(cum_bacbb.index, cum_bacbb.values, 'b-', linewidth=2, label='BACBB')
ax1.plot(cum_bacb.index, cum_bacb.values, 'r-', linewidth=2, label='BACB')
ax1.axvline(split_date, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='IS/OOS Split')
ax1.axvspan(bacbb.index[0], split_date, alpha=0.1, color='blue')
ax1.axvspan(split_date, bacbb.index[-1], alpha=0.1, color='red')

textstr_is = f'In-Sample\nBACBB: {m_bacbb_is["sharpe"]:.2f}\nBACB: {m_bacb_is["sharpe"]:.2f}'
textstr_oos = f'Out-of-Sample\nBACBB: {m_bacbb_oos["sharpe"]:.2f}\nBACB: {m_bacb_oos["sharpe"]:.2f}'
ax1.text(0.15, 0.95, textstr_is, transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax1.text(0.75, 0.95, textstr_oos, transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

ax1.set_title('전체 기간 누적 수익률 (샤프비율 표시)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Cumulative Return')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# 3-2: 연도별 성과 비교
ax2 = axes[0, 1]
years = bacbb.index.year.unique()
yearly_bacbb = []
yearly_bacb = []
yearly_labels = []

for year in years:
    mask = bacbb.index.year == year
    if mask.sum() > 50:
        yearly_bacbb.append(bacbb[mask].mean() * 252 * 100)
        yearly_bacb.append(bacb[mask].mean() * 252 * 100)
        yearly_labels.append(str(year))

x = np.arange(len(yearly_labels))
width = 0.35

bars1 = ax2.bar(x - width/2, yearly_bacbb, width, label='BACBB', color='blue', alpha=0.8)
bars2 = ax2.bar(x + width/2, yearly_bacb, width, label='BACB', color='red', alpha=0.8)

split_year_idx = yearly_labels.index('2023') if '2023' in yearly_labels else len(yearly_labels)//2
ax2.axvline(split_year_idx - 0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)

ax2.set_xticks(x)
ax2.set_xticklabels(yearly_labels, fontsize=11)
ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
ax2.set_title('연도별 수익률 비교', fontsize=13, fontweight='bold')
ax2.set_ylabel('Annual Return (%)')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# 3-3: Drawdown 비교
ax3 = axes[1, 0]
dd_bacbb = (cum_bacbb - cum_bacbb.cummax()) / cum_bacbb.cummax() * 100
dd_bacb = (cum_bacb - cum_bacb.cummax()) / cum_bacb.cummax() * 100

ax3.fill_between(dd_bacbb.index, dd_bacbb.values, 0, alpha=0.4, color='blue', label='BACBB')
ax3.fill_between(dd_bacb.index, dd_bacb.values, 0, alpha=0.4, color='red', label='BACB')
ax3.axvline(split_date, color='gray', linestyle='--', linewidth=2, alpha=0.7)

ax3.set_title('Drawdown 비교', fontsize=13, fontweight='bold')
ax3.set_ylabel('Drawdown (%)')
ax3.legend(loc='lower left', fontsize=10)
ax3.grid(True, alpha=0.3)

# 3-4: 성과 요약 테이블
ax4 = axes[1, 1]
ax4.axis('off')

table_data = [
    ['지표', 'BACBB IS', 'BACBB OOS', 'BACB IS', 'BACB OOS'],
    ['연수익률', f'{m_bacbb_is["ann_ret"]*100:.1f}%', f'{m_bacbb_oos["ann_ret"]*100:.1f}%', 
     f'{m_bacb_is["ann_ret"]*100:.1f}%', f'{m_bacb_oos["ann_ret"]*100:.1f}%'],
    ['변동성', f'{m_bacbb_is["ann_vol"]*100:.1f}%', f'{m_bacbb_oos["ann_vol"]*100:.1f}%',
     f'{m_bacb_is["ann_vol"]*100:.1f}%', f'{m_bacb_oos["ann_vol"]*100:.1f}%'],
    ['샤프비율', f'{m_bacbb_is["sharpe"]:.2f}', f'{m_bacbb_oos["sharpe"]:.2f}',
     f'{m_bacb_is["sharpe"]:.2f}', f'{m_bacb_oos["sharpe"]:.2f}'],
    ['소르티노', f'{m_bacbb_is["sortino"]:.2f}', f'{m_bacbb_oos["sortino"]:.2f}',
     f'{m_bacb_is["sortino"]:.2f}', f'{m_bacb_oos["sortino"]:.2f}'],
    ['MDD', f'{m_bacbb_is["mdd"]*100:.1f}%', f'{m_bacbb_oos["mdd"]*100:.1f}%',
     f'{m_bacb_is["mdd"]*100:.1f}%', f'{m_bacb_oos["mdd"]*100:.1f}%'],
    ['t-stat', f'{m_bacbb_is["t_stat"]:.2f}', f'{m_bacbb_oos["t_stat"]:.2f}',
     f'{m_bacb_is["t_stat"]:.2f}', f'{m_bacb_oos["t_stat"]:.2f}'],
    ['p-value', f'{m_bacbb_is["p_value"]:.4f}', f'{m_bacbb_oos["p_value"]:.4f}',
     f'{m_bacb_is["p_value"]:.4f}', f'{m_bacb_oos["p_value"]:.4f}'],
]

table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                  colWidths=[0.18, 0.18, 0.18, 0.18, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

for i in range(5):
    table[(0, i)].set_facecolor('#3949ab')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

for i in range(1, 8):
    table[(i, 1)].set_facecolor('#e3f2fd')
    table[(i, 2)].set_facecolor('#e3f2fd')

ax4.set_title('Out-of-Sample 검증 성과 요약', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('sample_15_OOS_Summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  sample_15_OOS_Summary.png 저장 완료")

print("\nOut-of-Sample 검증 그래프 생성 완료!")
