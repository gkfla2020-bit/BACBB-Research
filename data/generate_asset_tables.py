# -*- coding: utf-8 -*-
"""
고베타/저베타 자산별 성과 패널 표 생성
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("고베타/저베타 자산별 성과 표 생성")
print("="*70)

# 데이터 로드
prices = pd.read_csv('04_daily_prices.csv', index_col=0, parse_dates=True)
returns = pd.read_csv('06_daily_returns.csv', index_col=0, parse_dates=True)
volumes = pd.read_csv('05_daily_volumes.csv', index_col=0, parse_dates=True)

# 공통 자산
common = list(set(prices.columns) & set(returns.columns) & set(volumes.columns))
common = sorted(common)
print(f"분석 자산: {len(common)}개")

returns = returns[common].fillna(0).clip(-0.5, 0.5)

# 시장 수익률
vol_weights = volumes[common].div(volumes[common].sum(axis=1), axis=0).fillna(1/len(common))
market_ret = (returns * vol_weights).sum(axis=1)

# =============================================================================
# 베타 계산 (Frazzini-Pedersen 방식)
# =============================================================================
print("\n[베타 계산 중...]")

def calc_fp_beta(asset_ret, mkt_ret, corr_window=5, vol_window=252):
    """Frazzini-Pedersen 베타"""
    if len(asset_ret) < vol_window:
        return np.nan
    
    asset_vol = asset_ret.rolling(vol_window).std()
    mkt_vol = mkt_ret.rolling(vol_window).std()
    
    rolling_corr = asset_ret.rolling(corr_window).corr(mkt_ret)
    
    fp_beta = rolling_corr * (asset_vol / mkt_vol)
    return fp_beta.mean()

# 각 자산별 베타 및 성과 계산
asset_stats = []
for coin in common:
    r = returns[coin].dropna()
    if len(r) < 252:
        continue
    
    # 베타
    beta = calc_fp_beta(r, market_ret.loc[r.index])
    
    # 성과 지표
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # MDD
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    
    # t-stat
    t_stat = r.mean() / (r.std() / np.sqrt(len(r))) if r.std() > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(r)-1))
    
    asset_stats.append({
        'asset': coin,
        'beta': beta,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'mdd': mdd,
        't_stat': t_stat,
        'p_value': p_value,
        'n_obs': len(r)
    })

df = pd.DataFrame(asset_stats).sort_values('beta')
print(f"베타 계산 완료: {len(df)}개 자산")

# 상위/하위 분류
n = len(df)
n_group = max(n // 4, 5)  # 최소 5개

low_beta = df.head(n_group).copy()
high_beta = df.tail(n_group).copy()

print(f"\n저베타 자산: {len(low_beta)}개")
print(f"고베타 자산: {len(high_beta)}개")


# =============================================================================
# HTML 표 생성
# =============================================================================
print("\n[HTML 표 생성 중...]")

def get_sig(p):
    if p < 0.01: return '<span class="sig">***</span>'
    elif p < 0.05: return '<span class="sig">**</span>'
    elif p < 0.1: return '<span class="sig">*</span>'
    return ''

# 저베타 자산 행
low_rows = ""
for _, row in low_beta.iterrows():
    low_rows += f"""
                <tr>
                    <td class="left-align">{row['asset']}</td>
                    <td>{row['beta']:.3f}</td>
                    <td>{row['ann_ret']*100:.2f}{get_sig(row['p_value'])}</td>
                    <td>{row['ann_vol']*100:.2f}</td>
                    <td>{row['sharpe']:.2f}</td>
                    <td>{row['mdd']*100:.2f}</td>
                </tr>"""

# 고베타 자산 행
high_rows = ""
for _, row in high_beta.iterrows():
    high_rows += f"""
                <tr>
                    <td class="left-align">{row['asset']}</td>
                    <td>{row['beta']:.3f}</td>
                    <td>{row['ann_ret']*100:.2f}{get_sig(row['p_value'])}</td>
                    <td>{row['ann_vol']*100:.2f}</td>
                    <td>{row['sharpe']:.2f}</td>
                    <td>{row['mdd']*100:.2f}</td>
                </tr>"""

# 그룹 평균 (숫자 컬럼만)
numeric_cols = ['beta', 'ann_ret', 'ann_vol', 'sharpe', 'mdd', 't_stat', 'p_value', 'n_obs']
low_avg = low_beta[numeric_cols].mean()
high_avg = high_beta[numeric_cols].mean()

html_content = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>BACBB - Asset Beta Analysis</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Times New Roman', serif;
            background: #f5f5f5;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{ text-align: center; margin-bottom: 30px; color: #333; }}
        .table-container {{
            background: white;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .table-title {{
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
        }}
        table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        th, td {{ padding: 8px 12px; text-align: center; }}
        .top-rule {{ border-top: 2px solid black; }}
        .mid-rule {{ border-top: 1px solid black; }}
        .bottom-rule {{ border-bottom: 2px solid black; }}
        .panel-header {{ text-align: left; font-style: italic; padding-top: 15px; background: #f8f8f8; }}
        .left-align {{ text-align: left; }}
        .sig {{ vertical-align: super; font-size: 10px; color: #c00; }}
        .notes {{ font-size: 11px; color: #555; margin-top: 15px; text-align: left; font-style: italic; }}
        .highlight-low {{ background: #e6ffe6; }}
        .highlight-high {{ background: #ffe6e6; }}
        .avg-row {{ font-weight: bold; background: #f0f0f0; }}
        .summary-box {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card.low {{ border-left: 5px solid #28a745; }}
        .summary-card.high {{ border-left: 5px solid #dc3545; }}
        .summary-card h3 {{ margin-bottom: 15px; }}
        .summary-card .value {{ font-size: 28px; font-weight: bold; }}
        .summary-card .label {{ font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>BACBB Strategy: Beta Portfolio Composition</h1>
        
        <!-- 요약 카드 -->
        <div class="summary-box">
            <div class="summary-card low">
                <h3>🟢 Low Beta Portfolio (Long)</h3>
                <div class="value" style="color: #28a745;">{low_avg['beta']:.3f}</div>
                <div class="label">Average Beta</div>
                <hr style="margin: 15px 0;">
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 13px;">
                    <div><b>{low_avg['ann_ret']*100:.1f}%</b><br><span class="label">Annual Return</span></div>
                    <div><b>{low_avg['sharpe']:.2f}</b><br><span class="label">Sharpe Ratio</span></div>
                    <div><b>{low_avg['ann_vol']*100:.1f}%</b><br><span class="label">Volatility</span></div>
                    <div><b>{low_avg['mdd']*100:.1f}%</b><br><span class="label">Max DD</span></div>
                </div>
            </div>
            <div class="summary-card high">
                <h3>🔴 High Beta Portfolio (Short)</h3>
                <div class="value" style="color: #dc3545;">{high_avg['beta']:.3f}</div>
                <div class="label">Average Beta</div>
                <hr style="margin: 15px 0;">
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 13px;">
                    <div><b>{high_avg['ann_ret']*100:.1f}%</b><br><span class="label">Annual Return</span></div>
                    <div><b>{high_avg['sharpe']:.2f}</b><br><span class="label">Sharpe Ratio</span></div>
                    <div><b>{high_avg['ann_vol']*100:.1f}%</b><br><span class="label">Volatility</span></div>
                    <div><b>{high_avg['mdd']*100:.1f}%</b><br><span class="label">Max DD</span></div>
                </div>
            </div>
        </div>

        <!-- Table: 저베타 자산 -->
        <div class="table-container">
            <div class="table-title">Table 9: Low Beta Assets (Long Portfolio)</div>
            <table>
                <tr class="top-rule">
                    <th class="left-align">Asset</th>
                    <th>Beta (β)</th>
                    <th>Annual Return (%)</th>
                    <th>Volatility (%)</th>
                    <th>Sharpe Ratio</th>
                    <th>Max DD (%)</th>
                </tr>
                <tr class="mid-rule">
                    <td colspan="6" class="panel-header">Panel A: Individual Low Beta Assets</td>
                </tr>
                {low_rows}
                <tr class="mid-rule avg-row highlight-low">
                    <td class="left-align">Portfolio Average</td>
                    <td>{low_avg['beta']:.3f}</td>
                    <td>{low_avg['ann_ret']*100:.2f}</td>
                    <td>{low_avg['ann_vol']*100:.2f}</td>
                    <td>{low_avg['sharpe']:.2f}</td>
                    <td>{low_avg['mdd']*100:.2f}</td>
                </tr>
                <tr class="bottom-rule">
                    <td class="left-align">Number of Assets</td>
                    <td colspan="5">{len(low_beta)}</td>
                </tr>
            </table>
            <div class="notes">
                <i>Notes:</i> Low beta assets are selected from the bottom quartile of Frazzini-Pedersen beta estimates.
                These assets form the long leg of the BACBB strategy.
                <span class="sig">***</span>, <span class="sig">**</span>, <span class="sig">*</span> denote significance at 1%, 5%, 10% levels.
            </div>
        </div>

        <!-- Table: 고베타 자산 -->
        <div class="table-container">
            <div class="table-title">Table 10: High Beta Assets (Short Portfolio)</div>
            <table>
                <tr class="top-rule">
                    <th class="left-align">Asset</th>
                    <th>Beta (β)</th>
                    <th>Annual Return (%)</th>
                    <th>Volatility (%)</th>
                    <th>Sharpe Ratio</th>
                    <th>Max DD (%)</th>
                </tr>
                <tr class="mid-rule">
                    <td colspan="6" class="panel-header">Panel A: Individual High Beta Assets</td>
                </tr>
                {high_rows}
                <tr class="mid-rule avg-row highlight-high">
                    <td class="left-align">Portfolio Average</td>
                    <td>{high_avg['beta']:.3f}</td>
                    <td>{high_avg['ann_ret']*100:.2f}</td>
                    <td>{high_avg['ann_vol']*100:.2f}</td>
                    <td>{high_avg['sharpe']:.2f}</td>
                    <td>{high_avg['mdd']*100:.2f}</td>
                </tr>
                <tr class="bottom-rule">
                    <td class="left-align">Number of Assets</td>
                    <td colspan="5">{len(high_beta)}</td>
                </tr>
            </table>
            <div class="notes">
                <i>Notes:</i> High beta assets are selected from the top quartile of Frazzini-Pedersen beta estimates.
                These assets form the short leg of the BACBB strategy.
            </div>
        </div>

        <!-- Table: 비교 요약 -->
        <div class="table-container" style="border: 2px solid #0066cc;">
            <div class="table-title" style="color: #0066cc; font-size: 16px;">Table 11: Low Beta vs High Beta Portfolio Comparison</div>
            <table>
                <tr class="top-rule">
                    <th class="left-align">Metric</th>
                    <th>Low Beta (Long)</th>
                    <th>High Beta (Short)</th>
                    <th>Difference (L - H)</th>
                </tr>
                <tr class="mid-rule">
                    <td class="left-align">Average Beta</td>
                    <td class="highlight-low">{low_avg['beta']:.3f}</td>
                    <td class="highlight-high">{high_avg['beta']:.3f}</td>
                    <td>{low_avg['beta'] - high_avg['beta']:.3f}</td>
                </tr>
                <tr>
                    <td class="left-align">Annual Return (%)</td>
                    <td>{low_avg['ann_ret']*100:.2f}</td>
                    <td>{high_avg['ann_ret']*100:.2f}</td>
                    <td><b>{(low_avg['ann_ret'] - high_avg['ann_ret'])*100:+.2f}</b></td>
                </tr>
                <tr>
                    <td class="left-align">Volatility (%)</td>
                    <td>{low_avg['ann_vol']*100:.2f}</td>
                    <td>{high_avg['ann_vol']*100:.2f}</td>
                    <td>{(low_avg['ann_vol'] - high_avg['ann_vol'])*100:+.2f}</td>
                </tr>
                <tr>
                    <td class="left-align">Sharpe Ratio</td>
                    <td>{low_avg['sharpe']:.2f}</td>
                    <td>{high_avg['sharpe']:.2f}</td>
                    <td><b>{low_avg['sharpe'] - high_avg['sharpe']:+.2f}</b></td>
                </tr>
                <tr class="bottom-rule">
                    <td class="left-align">Max Drawdown (%)</td>
                    <td>{low_avg['mdd']*100:.2f}</td>
                    <td>{high_avg['mdd']*100:.2f}</td>
                    <td>{(low_avg['mdd'] - high_avg['mdd'])*100:+.2f}</td>
                </tr>
            </table>
            <div class="notes">
                <i>Notes:</i> This table compares the average characteristics of low beta and high beta portfolios.
                The BACBB strategy profits from the return spread between these two groups while hedging market exposure.
            </div>
        </div>

        <!-- 전체 자산 베타 분포 -->
        <div class="table-container">
            <div class="table-title">Table 12: Full Asset Beta Distribution</div>
            <table>
                <tr class="top-rule">
                    <th class="left-align">Statistic</th>
                    <th>Value</th>
                </tr>
                <tr class="mid-rule">
                    <td class="left-align">Total Assets</td>
                    <td>{len(df)}</td>
                </tr>
                <tr>
                    <td class="left-align">Mean Beta</td>
                    <td>{df['beta'].mean():.3f}</td>
                </tr>
                <tr>
                    <td class="left-align">Median Beta</td>
                    <td>{df['beta'].median():.3f}</td>
                </tr>
                <tr>
                    <td class="left-align">Std. Dev. Beta</td>
                    <td>{df['beta'].std():.3f}</td>
                </tr>
                <tr>
                    <td class="left-align">Min Beta</td>
                    <td>{df['beta'].min():.3f} ({df.iloc[0]['asset']})</td>
                </tr>
                <tr>
                    <td class="left-align">Max Beta</td>
                    <td>{df['beta'].max():.3f} ({df.iloc[-1]['asset']})</td>
                </tr>
                <tr>
                    <td class="left-align">25th Percentile</td>
                    <td>{df['beta'].quantile(0.25):.3f}</td>
                </tr>
                <tr class="bottom-rule">
                    <td class="left-align">75th Percentile</td>
                    <td>{df['beta'].quantile(0.75):.3f}</td>
                </tr>
            </table>
        </div>

    </div>
</body>
</html>
'''

with open('BACBB_Asset_Tables.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("\n" + "="*70)
print("완료!")
print("="*70)
print(f"저장: BACBB_Asset_Tables.html")
print(f"\n저베타 자산 (Long): {', '.join(low_beta['asset'].tolist())}")
print(f"\n고베타 자산 (Short): {', '.join(high_beta['asset'].tolist())}")
