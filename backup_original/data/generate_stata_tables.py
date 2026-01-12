# -*- coding: utf-8 -*-
"""
STATA 스타일 학술 논문용 LaTeX 표 생성
=====================================
- Panel A/B/C 형식의 다중 패널 표
- 유의수준 별표 표시 (*, **, ***)
- booktabs 스타일 (toprule, midrule, bottomrule)
- 소수점 정렬 및 표준오차 괄호 표시
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STATA 스타일 학술 논문용 LaTeX 표 생성")
print("="*70)

# =============================================================================
# 데이터 로드
# =============================================================================
print("\n[1. 데이터 로드]")

bacbb_returns = pd.read_csv('bacbb_returns.csv', index_col=0, parse_dates=True)['BACBB']
bacb_returns = pd.read_csv('bacb_returns.csv', index_col=0, parse_dates=True)['BACB']
bh_returns = pd.read_csv('bh_returns.csv', index_col=0, parse_dates=True)['BuyHold']

print(f"  BACBB: {len(bacbb_returns)} 관측치")
print(f"  BACB: {len(bacb_returns)} 관측치")
print(f"  Buy&Hold: {len(bh_returns)} 관측치")

# =============================================================================
# 유틸리티 함수
# =============================================================================
def get_significance_stars(p_value):
    """유의수준에 따른 별표 반환"""
    if p_value < 0.01:
        return "^{***}"
    elif p_value < 0.05:
        return "^{**}"
    elif p_value < 0.1:
        return "^{*}"
    return ""

def format_number(val, decimals=2, pct=False):
    """숫자 포맷팅"""
    if pd.isna(val):
        return ""
    if pct:
        return f"{val*100:.{decimals}f}"
    return f"{val:.{decimals}f}"

def calc_metrics(ret, name="Strategy"):
    """전략 성과 지표 계산"""
    ret = ret.dropna()
    if len(ret) < 50:
        return None
    
    n = len(ret)
    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    downside = ret[ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0
    
    cum = (1 + ret).cumprod()
    total_ret = cum.iloc[-1] - 1
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    
    win_rate = (ret > 0).mean()
    
    # t-통계량 및 p-value
    mean_ret = ret.mean()
    se = ret.std() / np.sqrt(n)
    t_stat = mean_ret / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-1))
    
    # 왜도, 첨도
    skewness = ret.skew()
    kurtosis = ret.kurtosis()
    
    return {
        'name': name,
        'n': n,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'total_ret': total_ret,
        'mdd': mdd,
        'win_rate': win_rate,
        't_stat': t_stat,
        'p_value': p_value,
        'se': se * np.sqrt(252),  # 연율화된 표준오차
        'skewness': skewness,
        'kurtosis': kurtosis
    }

# =============================================================================
# 성과 지표 계산
# =============================================================================
print("\n[2. 성과 지표 계산]")

m_bacbb = calc_metrics(bacbb_returns, "BACBB")
m_bacb = calc_metrics(bacb_returns, "BACB")
m_bh = calc_metrics(bh_returns, "Buy & Hold")

# In-Sample / Out-of-Sample 분할
split_idx = len(bacbb_returns) // 2
is_bacbb = bacbb_returns.iloc[:split_idx]
oos_bacbb = bacbb_returns.iloc[split_idx:]
is_bacb = bacb_returns.iloc[:split_idx]
oos_bacb = bacb_returns.iloc[split_idx:]

m_is_bacbb = calc_metrics(is_bacbb, "BACBB (IS)")
m_oos_bacbb = calc_metrics(oos_bacbb, "BACBB (OOS)")
m_is_bacb = calc_metrics(is_bacb, "BACB (IS)")
m_oos_bacb = calc_metrics(oos_bacb, "BACB (OOS)")

print(f"  BACBB: Sharpe={m_bacbb['sharpe']:.2f}, t={m_bacbb['t_stat']:.2f}")
print(f"  BACB: Sharpe={m_bacb['sharpe']:.2f}, t={m_bacb['t_stat']:.2f}")


# =============================================================================
# Table 1: 기술통계량 (Descriptive Statistics)
# =============================================================================
print("\n[3. Table 1: 기술통계량 생성]")

table1_latex = r"""
\begin{table}[htbp]
\centering
\caption{Descriptive Statistics of Daily Returns}
\label{tab:descriptive}
\begin{tabular}{l*{4}{c}}
\toprule
 & \multicolumn{1}{c}{BACBB} & \multicolumn{1}{c}{BACB} & \multicolumn{1}{c}{Buy \& Hold} & \multicolumn{1}{c}{Market} \\
\midrule
\multicolumn{5}{l}{\textit{Panel A: Return Statistics}} \\
\addlinespace
Mean (\%) & """ + f"{m_bacbb['ann_ret']*100:.2f}{get_significance_stars(m_bacbb['p_value'])}" + r""" & """ + f"{m_bacb['ann_ret']*100:.2f}{get_significance_stars(m_bacb['p_value'])}" + r""" & """ + f"{m_bh['ann_ret']*100:.2f}{get_significance_stars(m_bh['p_value'])}" + r""" & --- \\
 & """ + f"({m_bacbb['se']*100:.2f})" + r""" & """ + f"({m_bacb['se']*100:.2f})" + r""" & """ + f"({m_bh['se']*100:.2f})" + r""" & \\
Std. Dev. (\%) & """ + f"{m_bacbb['ann_vol']*100:.2f}" + r""" & """ + f"{m_bacb['ann_vol']*100:.2f}" + r""" & """ + f"{m_bh['ann_vol']*100:.2f}" + r""" & --- \\
Skewness & """ + f"{m_bacbb['skewness']:.3f}" + r""" & """ + f"{m_bacb['skewness']:.3f}" + r""" & """ + f"{m_bh['skewness']:.3f}" + r""" & --- \\
Kurtosis & """ + f"{m_bacbb['kurtosis']:.3f}" + r""" & """ + f"{m_bacb['kurtosis']:.3f}" + r""" & """ + f"{m_bh['kurtosis']:.3f}" + r""" & --- \\
\addlinespace
\multicolumn{5}{l}{\textit{Panel B: Risk-Adjusted Performance}} \\
\addlinespace
Sharpe Ratio & """ + f"{m_bacbb['sharpe']:.2f}" + r""" & """ + f"{m_bacb['sharpe']:.2f}" + r""" & """ + f"{m_bh['sharpe']:.2f}" + r""" & --- \\
Sortino Ratio & """ + f"{m_bacbb['sortino']:.2f}" + r""" & """ + f"{m_bacb['sortino']:.2f}" + r""" & """ + f"{m_bh['sortino']:.2f}" + r""" & --- \\
Calmar Ratio & """ + f"{m_bacbb['calmar']:.2f}" + r""" & """ + f"{m_bacb['calmar']:.2f}" + r""" & """ + f"{m_bh['calmar']:.2f}" + r""" & --- \\
\addlinespace
\multicolumn{5}{l}{\textit{Panel C: Drawdown \& Win Rate}} \\
\addlinespace
Max Drawdown (\%) & """ + f"{m_bacbb['mdd']*100:.2f}" + r""" & """ + f"{m_bacb['mdd']*100:.2f}" + r""" & """ + f"{m_bh['mdd']*100:.2f}" + r""" & --- \\
Win Rate (\%) & """ + f"{m_bacbb['win_rate']*100:.1f}" + r""" & """ + f"{m_bacb['win_rate']*100:.1f}" + r""" & """ + f"{m_bh['win_rate']*100:.1f}" + r""" & --- \\
\addlinespace
\multicolumn{5}{l}{\textit{Panel D: Statistical Significance}} \\
\addlinespace
$t$-statistic & """ + f"{m_bacbb['t_stat']:.2f}" + r""" & """ + f"{m_bacb['t_stat']:.2f}" + r""" & """ + f"{m_bh['t_stat']:.2f}" + r""" & --- \\
$p$-value & """ + f"{m_bacbb['p_value']:.4f}" + r""" & """ + f"{m_bacb['p_value']:.4f}" + r""" & """ + f"{m_bh['p_value']:.4f}" + r""" & --- \\
Observations & """ + f"{m_bacbb['n']:,}" + r""" & """ + f"{m_bacb['n']:,}" + r""" & """ + f"{m_bh['n']:,}" + r""" & --- \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} This table reports descriptive statistics for daily returns of BACBB (Betting Against Cryptocurrency Bad Beta), BACB (Betting Against Cryptocurrency Beta), and Buy \& Hold strategies. Mean and Std. Dev. are annualized. Standard errors are in parentheses. $^{***}$, $^{**}$, and $^{*}$ denote statistical significance at the 1\%, 5\%, and 10\% levels, respectively.
\end{tablenotes}
\end{table}
"""

with open('Table_1_Descriptive_Stats.tex', 'w', encoding='utf-8') as f:
    f.write(table1_latex)
print("  Table 1 저장: Table_1_Descriptive_Stats.tex")


# =============================================================================
# Table 2: Out-of-Sample 검증 결과
# =============================================================================
print("\n[4. Table 2: Out-of-Sample 검증 결과 생성]")

table2_latex = r"""
\begin{table}[htbp]
\centering
\caption{Out-of-Sample Validation Results}
\label{tab:oos}
\begin{tabular}{l*{4}{c}}
\toprule
 & \multicolumn{2}{c}{BACBB} & \multicolumn{2}{c}{BACB} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
 & In-Sample & Out-of-Sample & In-Sample & Out-of-Sample \\
\midrule
\multicolumn{5}{l}{\textit{Panel A: Return Performance}} \\
\addlinespace
Annual Return (\%) & """ + f"{m_is_bacbb['ann_ret']*100:.2f}{get_significance_stars(m_is_bacbb['p_value'])}" + r""" & """ + f"{m_oos_bacbb['ann_ret']*100:.2f}{get_significance_stars(m_oos_bacbb['p_value'])}" + r""" & """ + f"{m_is_bacb['ann_ret']*100:.2f}{get_significance_stars(m_is_bacb['p_value'])}" + r""" & """ + f"{m_oos_bacb['ann_ret']*100:.2f}{get_significance_stars(m_oos_bacb['p_value'])}" + r""" \\
 & """ + f"({m_is_bacbb['se']*100:.2f})" + r""" & """ + f"({m_oos_bacbb['se']*100:.2f})" + r""" & """ + f"({m_is_bacb['se']*100:.2f})" + r""" & """ + f"({m_oos_bacb['se']*100:.2f})" + r""" \\
Volatility (\%) & """ + f"{m_is_bacbb['ann_vol']*100:.2f}" + r""" & """ + f"{m_oos_bacbb['ann_vol']*100:.2f}" + r""" & """ + f"{m_is_bacb['ann_vol']*100:.2f}" + r""" & """ + f"{m_oos_bacb['ann_vol']*100:.2f}" + r""" \\
\addlinespace
\multicolumn{5}{l}{\textit{Panel B: Risk-Adjusted Metrics}} \\
\addlinespace
Sharpe Ratio & """ + f"{m_is_bacbb['sharpe']:.2f}" + r""" & """ + f"{m_oos_bacbb['sharpe']:.2f}" + r""" & """ + f"{m_is_bacb['sharpe']:.2f}" + r""" & """ + f"{m_oos_bacb['sharpe']:.2f}" + r""" \\
Sortino Ratio & """ + f"{m_is_bacbb['sortino']:.2f}" + r""" & """ + f"{m_oos_bacbb['sortino']:.2f}" + r""" & """ + f"{m_is_bacb['sortino']:.2f}" + r""" & """ + f"{m_oos_bacb['sortino']:.2f}" + r""" \\
Max Drawdown (\%) & """ + f"{m_is_bacbb['mdd']*100:.2f}" + r""" & """ + f"{m_oos_bacbb['mdd']*100:.2f}" + r""" & """ + f"{m_is_bacb['mdd']*100:.2f}" + r""" & """ + f"{m_oos_bacb['mdd']*100:.2f}" + r""" \\
\addlinespace
\multicolumn{5}{l}{\textit{Panel C: Statistical Tests}} \\
\addlinespace
$t$-statistic & """ + f"{m_is_bacbb['t_stat']:.2f}" + r""" & """ + f"{m_oos_bacbb['t_stat']:.2f}" + r""" & """ + f"{m_is_bacb['t_stat']:.2f}" + r""" & """ + f"{m_oos_bacb['t_stat']:.2f}" + r""" \\
$p$-value & """ + f"{m_is_bacbb['p_value']:.4f}" + r""" & """ + f"{m_oos_bacbb['p_value']:.4f}" + r""" & """ + f"{m_is_bacb['p_value']:.4f}" + r""" & """ + f"{m_oos_bacb['p_value']:.4f}" + r""" \\
Observations & """ + f"{m_is_bacbb['n']:,}" + r""" & """ + f"{m_oos_bacbb['n']:,}" + r""" & """ + f"{m_is_bacb['n']:,}" + r""" & """ + f"{m_oos_bacb['n']:,}" + r""" \\
\addlinespace
Period & """ + f"{is_bacbb.index[0].strftime('%Y-%m')}" + r"""--""" + f"{is_bacbb.index[-1].strftime('%Y-%m')}" + r""" & """ + f"{oos_bacbb.index[0].strftime('%Y-%m')}" + r"""--""" + f"{oos_bacbb.index[-1].strftime('%Y-%m')}" + r""" & """ + f"{is_bacb.index[0].strftime('%Y-%m')}" + r"""--""" + f"{is_bacb.index[-1].strftime('%Y-%m')}" + r""" & """ + f"{oos_bacb.index[0].strftime('%Y-%m')}" + r"""--""" + f"{oos_bacb.index[-1].strftime('%Y-%m')}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} This table presents out-of-sample validation results. The sample is split at the midpoint. Standard errors are in parentheses. $^{***}$, $^{**}$, and $^{*}$ denote statistical significance at the 1\%, 5\%, and 10\% levels, respectively.
\end{tablenotes}
\end{table}
"""

with open('Table_2_OOS_Validation.tex', 'w', encoding='utf-8') as f:
    f.write(table2_latex)
print("  Table 2 저장: Table_2_OOS_Validation.tex")


# =============================================================================
# Table 3: 연도별 성과 (Yearly Performance)
# =============================================================================
print("\n[5. Table 3: 연도별 성과 생성]")

# 연도별 수익률 계산
yearly_bacbb = bacbb_returns.groupby(bacbb_returns.index.year).apply(lambda x: (1+x).prod()-1)
yearly_bacb = bacb_returns.groupby(bacb_returns.index.year).apply(lambda x: (1+x).prod()-1)
yearly_bh = bh_returns.groupby(bh_returns.index.year).apply(lambda x: (1+x).prod()-1)

# 연도별 샤프비율
yearly_sharpe_bacbb = bacbb_returns.groupby(bacbb_returns.index.year).apply(
    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0)
yearly_sharpe_bacb = bacb_returns.groupby(bacb_returns.index.year).apply(
    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0)

years = sorted(set(yearly_bacbb.index) & set(yearly_bacb.index) & set(yearly_bh.index))

table3_rows = ""
for year in years:
    ret_bacbb = yearly_bacbb.get(year, 0) * 100
    ret_bacb = yearly_bacb.get(year, 0) * 100
    ret_bh = yearly_bh.get(year, 0) * 100
    sr_bacbb = yearly_sharpe_bacbb.get(year, 0)
    sr_bacb = yearly_sharpe_bacb.get(year, 0)
    
    table3_rows += f"{year} & {ret_bacbb:.2f} & {sr_bacbb:.2f} & {ret_bacb:.2f} & {sr_bacb:.2f} & {ret_bh:.2f} \\\\\n"

# 전체 기간 평균
avg_ret_bacbb = yearly_bacbb.mean() * 100
avg_ret_bacb = yearly_bacb.mean() * 100
avg_ret_bh = yearly_bh.mean() * 100
avg_sr_bacbb = yearly_sharpe_bacbb.mean()
avg_sr_bacb = yearly_sharpe_bacb.mean()

table3_latex = r"""
\begin{table}[htbp]
\centering
\caption{Annual Performance by Year}
\label{tab:yearly}
\begin{tabular}{l*{5}{c}}
\toprule
 & \multicolumn{2}{c}{BACBB} & \multicolumn{2}{c}{BACB} & Buy \& Hold \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-6}
Year & Return (\%) & Sharpe & Return (\%) & Sharpe & Return (\%) \\
\midrule
""" + table3_rows + r"""\midrule
Average & """ + f"{avg_ret_bacbb:.2f}" + r""" & """ + f"{avg_sr_bacbb:.2f}" + r""" & """ + f"{avg_ret_bacb:.2f}" + r""" & """ + f"{avg_sr_bacb:.2f}" + r""" & """ + f"{avg_ret_bh:.2f}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} This table reports annual returns and Sharpe ratios for each strategy by calendar year. Returns are expressed in percentage terms.
\end{tablenotes}
\end{table}
"""

with open('Table_3_Yearly_Performance.tex', 'w', encoding='utf-8') as f:
    f.write(table3_latex)
print("  Table 3 저장: Table_3_Yearly_Performance.tex")


# =============================================================================
# Table 4: 회귀분석 결과 (Regression Results) - STATA estout 스타일
# =============================================================================
print("\n[6. Table 4: 회귀분석 결과 생성]")

# 공통 인덱스
common_idx = bacbb_returns.index.intersection(bacb_returns.index).intersection(bh_returns.index)
bacbb_aligned = bacbb_returns.loc[common_idx]
bacb_aligned = bacb_returns.loc[common_idx]
bh_aligned = bh_returns.loc[common_idx]

# 회귀분석 1: BACBB on Market
slope1, intercept1, r1, p1, se1 = stats.linregress(bh_aligned, bacbb_aligned)
n1 = len(common_idx)
t_alpha1 = intercept1 / (bacbb_aligned.std() / np.sqrt(n1))
p_alpha1 = 2 * (1 - stats.t.cdf(abs(t_alpha1), n1-2))
t_beta1 = slope1 / se1
p_beta1 = 2 * (1 - stats.t.cdf(abs(t_beta1), n1-2))

# 회귀분석 2: BACB on Market
slope2, intercept2, r2, p2, se2 = stats.linregress(bh_aligned, bacb_aligned)
t_alpha2 = intercept2 / (bacb_aligned.std() / np.sqrt(n1))
p_alpha2 = 2 * (1 - stats.t.cdf(abs(t_alpha2), n1-2))
t_beta2 = slope2 / se2
p_beta2 = 2 * (1 - stats.t.cdf(abs(t_beta2), n1-2))

# 회귀분석 3: BACBB on BACB
slope3, intercept3, r3, p3, se3 = stats.linregress(bacb_aligned, bacbb_aligned)
t_alpha3 = intercept3 / (bacbb_aligned.std() / np.sqrt(n1))
p_alpha3 = 2 * (1 - stats.t.cdf(abs(t_alpha3), n1-2))
t_beta3 = slope3 / se3
p_beta3 = 2 * (1 - stats.t.cdf(abs(t_beta3), n1-2))

# 연율화된 알파
alpha1_ann = intercept1 * 252 * 100
alpha2_ann = intercept2 * 252 * 100
alpha3_ann = intercept3 * 252 * 100

table4_latex = r"""
\begin{table}[htbp]
\centering
\caption{Factor Regression Results}
\label{tab:regression}
\begin{tabular}{l*{3}{c}}
\toprule
 & \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)} & \multicolumn{1}{c}{(3)} \\
 & BACBB & BACB & BACBB \\
\midrule
\multicolumn{4}{l}{\textit{Panel A: Market Model}} \\
\addlinespace
$\alpha$ (annualized, \%) & """ + f"{alpha1_ann:.2f}{get_significance_stars(p_alpha1)}" + r""" & """ + f"{alpha2_ann:.2f}{get_significance_stars(p_alpha2)}" + r""" & """ + f"{alpha3_ann:.2f}{get_significance_stars(p_alpha3)}" + r""" \\
 & """ + f"({abs(t_alpha1):.2f})" + r""" & """ + f"({abs(t_alpha2):.2f})" + r""" & """ + f"({abs(t_alpha3):.2f})" + r""" \\
$\beta_{Market}$ & """ + f"{slope1:.3f}{get_significance_stars(p_beta1)}" + r""" & """ + f"{slope2:.3f}{get_significance_stars(p_beta2)}" + r""" & --- \\
 & """ + f"({abs(t_beta1):.2f})" + r""" & """ + f"({abs(t_beta2):.2f})" + r""" & \\
$\beta_{BACB}$ & --- & --- & """ + f"{slope3:.3f}{get_significance_stars(p_beta3)}" + r""" \\
 & & & """ + f"({abs(t_beta3):.2f})" + r""" \\
\addlinespace
\multicolumn{4}{l}{\textit{Panel B: Model Statistics}} \\
\addlinespace
$R^2$ & """ + f"{r1**2:.3f}" + r""" & """ + f"{r2**2:.3f}" + r""" & """ + f"{r3**2:.3f}" + r""" \\
Adjusted $R^2$ & """ + f"{1-(1-r1**2)*(n1-1)/(n1-2):.3f}" + r""" & """ + f"{1-(1-r2**2)*(n1-1)/(n1-2):.3f}" + r""" & """ + f"{1-(1-r3**2)*(n1-1)/(n1-2):.3f}" + r""" \\
Observations & """ + f"{n1:,}" + r""" & """ + f"{n1:,}" + r""" & """ + f"{n1:,}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} This table reports OLS regression results. Columns (1) and (2) regress strategy returns on market returns. Column (3) regresses BACBB on BACB. $t$-statistics are in parentheses. $^{***}$, $^{**}$, and $^{*}$ denote statistical significance at the 1\%, 5\%, and 10\% levels, respectively.
\end{tablenotes}
\end{table}
"""

with open('Table_4_Regression.tex', 'w', encoding='utf-8') as f:
    f.write(table4_latex)
print("  Table 4 저장: Table_4_Regression.tex")


# =============================================================================
# Table 5: 하락장 방어력 분석 (Downside Protection)
# =============================================================================
print("\n[7. Table 5: 하락장 방어력 분석 생성]")

# 시장 하락일 정의 (2% 이상 하락)
down_days = bh_aligned[bh_aligned < -0.02].index
up_days = bh_aligned[bh_aligned > 0.02].index
normal_days = bh_aligned[(bh_aligned >= -0.02) & (bh_aligned <= 0.02)].index

# 각 시장 상황별 성과
def calc_conditional_stats(ret, condition_idx):
    r = ret.loc[condition_idx]
    if len(r) < 10:
        return {'mean': np.nan, 'std': np.nan, 'n': 0}
    return {
        'mean': r.mean() * 100,  # 일간 수익률 %
        'std': r.std() * 100,
        'n': len(r)
    }

down_bacbb = calc_conditional_stats(bacbb_aligned, down_days)
down_bacb = calc_conditional_stats(bacb_aligned, down_days)
down_bh = calc_conditional_stats(bh_aligned, down_days)

up_bacbb = calc_conditional_stats(bacbb_aligned, up_days)
up_bacb = calc_conditional_stats(bacb_aligned, up_days)
up_bh = calc_conditional_stats(bh_aligned, up_days)

normal_bacbb = calc_conditional_stats(bacbb_aligned, normal_days)
normal_bacb = calc_conditional_stats(bacb_aligned, normal_days)
normal_bh = calc_conditional_stats(bh_aligned, normal_days)

# 방어율 계산
defense_bacbb = 1 - (down_bacbb['mean'] / down_bh['mean']) if down_bh['mean'] != 0 else 0
defense_bacb = 1 - (down_bacb['mean'] / down_bh['mean']) if down_bh['mean'] != 0 else 0

table5_latex = r"""
\begin{table}[htbp]
\centering
\caption{Conditional Performance Analysis}
\label{tab:conditional}
\begin{tabular}{l*{3}{c}}
\toprule
 & \multicolumn{1}{c}{BACBB} & \multicolumn{1}{c}{BACB} & \multicolumn{1}{c}{Market} \\
\midrule
\multicolumn{4}{l}{\textit{Panel A: Market Down Days ($r_m < -2\%$)}} \\
\addlinespace
Mean Return (\%) & """ + f"{down_bacbb['mean']:.3f}" + r""" & """ + f"{down_bacb['mean']:.3f}" + r""" & """ + f"{down_bh['mean']:.3f}" + r""" \\
Std. Dev. (\%) & """ + f"{down_bacbb['std']:.3f}" + r""" & """ + f"{down_bacb['std']:.3f}" + r""" & """ + f"{down_bh['std']:.3f}" + r""" \\
Observations & """ + f"{down_bacbb['n']}" + r""" & """ + f"{down_bacb['n']}" + r""" & """ + f"{down_bh['n']}" + r""" \\
Defense Rate (\%) & """ + f"{defense_bacbb*100:.1f}" + r""" & """ + f"{defense_bacb*100:.1f}" + r""" & --- \\
\addlinespace
\multicolumn{4}{l}{\textit{Panel B: Market Up Days ($r_m > +2\%$)}} \\
\addlinespace
Mean Return (\%) & """ + f"{up_bacbb['mean']:.3f}" + r""" & """ + f"{up_bacb['mean']:.3f}" + r""" & """ + f"{up_bh['mean']:.3f}" + r""" \\
Std. Dev. (\%) & """ + f"{up_bacbb['std']:.3f}" + r""" & """ + f"{up_bacb['std']:.3f}" + r""" & """ + f"{up_bh['std']:.3f}" + r""" \\
Observations & """ + f"{up_bacbb['n']}" + r""" & """ + f"{up_bacb['n']}" + r""" & """ + f"{up_bh['n']}" + r""" \\
\addlinespace
\multicolumn{4}{l}{\textit{Panel C: Normal Days ($-2\% \leq r_m \leq +2\%$)}} \\
\addlinespace
Mean Return (\%) & """ + f"{normal_bacbb['mean']:.3f}" + r""" & """ + f"{normal_bacb['mean']:.3f}" + r""" & """ + f"{normal_bh['mean']:.3f}" + r""" \\
Std. Dev. (\%) & """ + f"{normal_bacbb['std']:.3f}" + r""" & """ + f"{normal_bacb['std']:.3f}" + r""" & """ + f"{normal_bh['std']:.3f}" + r""" \\
Observations & """ + f"{normal_bacbb['n']}" + r""" & """ + f"{normal_bacb['n']}" + r""" & """ + f"{normal_bh['n']}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} This table reports conditional performance statistics based on market return regimes. Defense Rate measures the percentage of market losses avoided during down days.
\end{tablenotes}
\end{table}
"""

with open('Table_5_Conditional_Performance.tex', 'w', encoding='utf-8') as f:
    f.write(table5_latex)
print("  Table 5 저장: Table_5_Conditional_Performance.tex")


# =============================================================================
# Table 6: 롤링 윈도우 분석 (Rolling Window Analysis)
# =============================================================================
print("\n[8. Table 6: 롤링 윈도우 분석 생성]")

# 다양한 윈도우 크기로 롤링 샤프비율 계산
windows = [63, 126, 252, 504]  # 3개월, 6개월, 1년, 2년

rolling_stats = []
for w in windows:
    rs_bacbb = bacbb_returns.rolling(w).mean() / bacbb_returns.rolling(w).std() * np.sqrt(252)
    rs_bacb = bacb_returns.rolling(w).mean() / bacb_returns.rolling(w).std() * np.sqrt(252)
    
    rolling_stats.append({
        'window': w,
        'bacbb_mean': rs_bacbb.mean(),
        'bacbb_std': rs_bacbb.std(),
        'bacbb_min': rs_bacbb.min(),
        'bacbb_max': rs_bacbb.max(),
        'bacbb_pct_positive': (rs_bacbb > 0).mean() * 100,
        'bacb_mean': rs_bacb.mean(),
        'bacb_std': rs_bacb.std(),
        'bacb_min': rs_bacb.min(),
        'bacb_max': rs_bacb.max(),
        'bacb_pct_positive': (rs_bacb > 0).mean() * 100,
    })

table6_rows = ""
window_labels = ['3-Month', '6-Month', '1-Year', '2-Year']
for i, (w, label) in enumerate(zip(windows, window_labels)):
    s = rolling_stats[i]
    table6_rows += f"{label} ({w}d) & {s['bacbb_mean']:.2f} & {s['bacbb_std']:.2f} & {s['bacbb_pct_positive']:.1f} & {s['bacb_mean']:.2f} & {s['bacb_std']:.2f} & {s['bacb_pct_positive']:.1f} \\\\\n"

table6_latex = r"""
\begin{table}[htbp]
\centering
\caption{Rolling Window Sharpe Ratio Analysis}
\label{tab:rolling}
\begin{tabular}{l*{6}{c}}
\toprule
 & \multicolumn{3}{c}{BACBB} & \multicolumn{3}{c}{BACB} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
Window & Mean & Std. Dev. & \% Positive & Mean & Std. Dev. & \% Positive \\
\midrule
""" + table6_rows + r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} This table reports statistics of rolling Sharpe ratios computed over different window lengths. \% Positive indicates the percentage of observations with positive Sharpe ratios.
\end{tablenotes}
\end{table}
"""

with open('Table_6_Rolling_Analysis.tex', 'w', encoding='utf-8') as f:
    f.write(table6_latex)
print("  Table 6 저장: Table_6_Rolling_Analysis.tex")


# =============================================================================
# Table 7: 종합 성과 비교표 (Summary Comparison) - 가장 중요한 표
# =============================================================================
print("\n[9. Table 7: 종합 성과 비교표 생성]")

# 상관계수 계산
corr_bacbb_bacb = bacbb_aligned.corr(bacb_aligned)
corr_bacbb_mkt = bacbb_aligned.corr(bh_aligned)
corr_bacb_mkt = bacb_aligned.corr(bh_aligned)

# 정보비율 (Information Ratio)
tracking_error_bacbb = (bacbb_aligned - bh_aligned).std() * np.sqrt(252)
tracking_error_bacb = (bacb_aligned - bh_aligned).std() * np.sqrt(252)
ir_bacbb = (m_bacbb['ann_ret'] - m_bh['ann_ret']) / tracking_error_bacbb if tracking_error_bacbb > 0 else 0
ir_bacb = (m_bacb['ann_ret'] - m_bh['ann_ret']) / tracking_error_bacb if tracking_error_bacb > 0 else 0

table7_latex = r"""
\begin{table}[htbp]
\centering
\caption{Comprehensive Performance Comparison}
\label{tab:summary}
\begin{tabular}{l*{3}{c}}
\toprule
Metric & BACBB & BACB & Buy \& Hold \\
\midrule
\multicolumn{4}{l}{\textit{Panel A: Return Metrics}} \\
\addlinespace
Annual Return (\%) & """ + f"{m_bacbb['ann_ret']*100:.2f}{get_significance_stars(m_bacbb['p_value'])}" + r""" & """ + f"{m_bacb['ann_ret']*100:.2f}{get_significance_stars(m_bacb['p_value'])}" + r""" & """ + f"{m_bh['ann_ret']*100:.2f}{get_significance_stars(m_bh['p_value'])}" + r""" \\
Total Return (\%) & """ + f"{m_bacbb['total_ret']*100:.2f}" + r""" & """ + f"{m_bacb['total_ret']*100:.2f}" + r""" & """ + f"{m_bh['total_ret']*100:.2f}" + r""" \\
Annual Volatility (\%) & """ + f"{m_bacbb['ann_vol']*100:.2f}" + r""" & """ + f"{m_bacb['ann_vol']*100:.2f}" + r""" & """ + f"{m_bh['ann_vol']*100:.2f}" + r""" \\
\addlinespace
\multicolumn{4}{l}{\textit{Panel B: Risk-Adjusted Performance}} \\
\addlinespace
Sharpe Ratio & """ + f"{m_bacbb['sharpe']:.2f}" + r""" & """ + f"{m_bacb['sharpe']:.2f}" + r""" & """ + f"{m_bh['sharpe']:.2f}" + r""" \\
Sortino Ratio & """ + f"{m_bacbb['sortino']:.2f}" + r""" & """ + f"{m_bacb['sortino']:.2f}" + r""" & """ + f"{m_bh['sortino']:.2f}" + r""" \\
Calmar Ratio & """ + f"{m_bacbb['calmar']:.2f}" + r""" & """ + f"{m_bacb['calmar']:.2f}" + r""" & """ + f"{m_bh['calmar']:.2f}" + r""" \\
Information Ratio & """ + f"{ir_bacbb:.2f}" + r""" & """ + f"{ir_bacb:.2f}" + r""" & --- \\
\addlinespace
\multicolumn{4}{l}{\textit{Panel C: Risk Metrics}} \\
\addlinespace
Max Drawdown (\%) & """ + f"{m_bacbb['mdd']*100:.2f}" + r""" & """ + f"{m_bacb['mdd']*100:.2f}" + r""" & """ + f"{m_bh['mdd']*100:.2f}" + r""" \\
Win Rate (\%) & """ + f"{m_bacbb['win_rate']*100:.1f}" + r""" & """ + f"{m_bacb['win_rate']*100:.1f}" + r""" & """ + f"{m_bh['win_rate']*100:.1f}" + r""" \\
Skewness & """ + f"{m_bacbb['skewness']:.3f}" + r""" & """ + f"{m_bacb['skewness']:.3f}" + r""" & """ + f"{m_bh['skewness']:.3f}" + r""" \\
Kurtosis & """ + f"{m_bacbb['kurtosis']:.3f}" + r""" & """ + f"{m_bacb['kurtosis']:.3f}" + r""" & """ + f"{m_bh['kurtosis']:.3f}" + r""" \\
\addlinespace
\multicolumn{4}{l}{\textit{Panel D: Correlations}} \\
\addlinespace
Corr. with Market & """ + f"{corr_bacbb_mkt:.3f}" + r""" & """ + f"{corr_bacb_mkt:.3f}" + r""" & 1.000 \\
Corr. BACBB-BACB & \multicolumn{2}{c}{""" + f"{corr_bacbb_bacb:.3f}" + r"""} & --- \\
\addlinespace
\multicolumn{4}{l}{\textit{Panel E: Statistical Tests}} \\
\addlinespace
$t$-statistic & """ + f"{m_bacbb['t_stat']:.2f}" + r""" & """ + f"{m_bacb['t_stat']:.2f}" + r""" & """ + f"{m_bh['t_stat']:.2f}" + r""" \\
$p$-value & """ + f"{m_bacbb['p_value']:.4f}" + r""" & """ + f"{m_bacb['p_value']:.4f}" + r""" & """ + f"{m_bh['p_value']:.4f}" + r""" \\
Observations & """ + f"{m_bacbb['n']:,}" + r""" & """ + f"{m_bacb['n']:,}" + r""" & """ + f"{m_bh['n']:,}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} This table provides a comprehensive comparison of strategy performance. Annual Return and Volatility are annualized from daily returns. Information Ratio is calculated relative to the market benchmark. $^{***}$, $^{**}$, and $^{*}$ denote statistical significance at the 1\%, 5\%, and 10\% levels, respectively.
\end{tablenotes}
\end{table}
"""

with open('Table_7_Summary_Comparison.tex', 'w', encoding='utf-8') as f:
    f.write(table7_latex)
print("  Table 7 저장: Table_7_Summary_Comparison.tex")


# =============================================================================
# Table 8: Robustness Check - 다양한 기간별 성과
# =============================================================================
print("\n[10. Table 8: Robustness Check 생성]")

# 분기별 성과
quarterly_bacbb = bacbb_returns.resample('Q').apply(lambda x: (1+x).prod()-1)
quarterly_bacb = bacb_returns.resample('Q').apply(lambda x: (1+x).prod()-1)

# 분기별 승률
q_win_bacbb = (quarterly_bacbb > 0).mean() * 100
q_win_bacb = (quarterly_bacb > 0).mean() * 100

# 연속 양수/음수 분기
def max_consecutive(series, positive=True):
    if positive:
        mask = series > 0
    else:
        mask = series < 0
    groups = (mask != mask.shift()).cumsum()
    return mask.groupby(groups).sum().max()

max_pos_q_bacbb = max_consecutive(quarterly_bacbb, True)
max_neg_q_bacbb = max_consecutive(quarterly_bacbb, False)
max_pos_q_bacb = max_consecutive(quarterly_bacb, True)
max_neg_q_bacb = max_consecutive(quarterly_bacb, False)

# 월별 성과
monthly_bacbb = bacbb_returns.resample('M').apply(lambda x: (1+x).prod()-1)
monthly_bacb = bacb_returns.resample('M').apply(lambda x: (1+x).prod()-1)

m_win_bacbb = (monthly_bacbb > 0).mean() * 100
m_win_bacb = (monthly_bacb > 0).mean() * 100

# 최고/최저 월
best_month_bacbb = monthly_bacbb.max() * 100
worst_month_bacbb = monthly_bacbb.min() * 100
best_month_bacb = monthly_bacb.max() * 100
worst_month_bacb = monthly_bacb.min() * 100

table8_latex = r"""
\begin{table}[htbp]
\centering
\caption{Robustness Analysis: Performance Across Different Horizons}
\label{tab:robustness}
\begin{tabular}{l*{2}{c}}
\toprule
Metric & BACBB & BACB \\
\midrule
\multicolumn{3}{l}{\textit{Panel A: Monthly Statistics}} \\
\addlinespace
Monthly Win Rate (\%) & """ + f"{m_win_bacbb:.1f}" + r""" & """ + f"{m_win_bacb:.1f}" + r""" \\
Best Month (\%) & """ + f"{best_month_bacbb:.2f}" + r""" & """ + f"{best_month_bacb:.2f}" + r""" \\
Worst Month (\%) & """ + f"{worst_month_bacbb:.2f}" + r""" & """ + f"{worst_month_bacb:.2f}" + r""" \\
Number of Months & """ + f"{len(monthly_bacbb)}" + r""" & """ + f"{len(monthly_bacb)}" + r""" \\
\addlinespace
\multicolumn{3}{l}{\textit{Panel B: Quarterly Statistics}} \\
\addlinespace
Quarterly Win Rate (\%) & """ + f"{q_win_bacbb:.1f}" + r""" & """ + f"{q_win_bacb:.1f}" + r""" \\
Max Consecutive Positive Quarters & """ + f"{max_pos_q_bacbb:.0f}" + r""" & """ + f"{max_pos_q_bacb:.0f}" + r""" \\
Max Consecutive Negative Quarters & """ + f"{max_neg_q_bacbb:.0f}" + r""" & """ + f"{max_neg_q_bacb:.0f}" + r""" \\
Number of Quarters & """ + f"{len(quarterly_bacbb)}" + r""" & """ + f"{len(quarterly_bacb)}" + r""" \\
\addlinespace
\multicolumn{3}{l}{\textit{Panel C: Annual Statistics}} \\
\addlinespace
Annual Win Rate (\%) & """ + f"{(yearly_bacbb > 0).mean()*100:.1f}" + r""" & """ + f"{(yearly_bacb > 0).mean()*100:.1f}" + r""" \\
Best Year (\%) & """ + f"{yearly_bacbb.max()*100:.2f}" + r""" & """ + f"{yearly_bacb.max()*100:.2f}" + r""" \\
Worst Year (\%) & """ + f"{yearly_bacbb.min()*100:.2f}" + r""" & """ + f"{yearly_bacb.min()*100:.2f}" + r""" \\
Number of Years & """ + f"{len(yearly_bacbb)}" + r""" & """ + f"{len(yearly_bacb)}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} This table reports performance statistics across different time horizons to assess strategy robustness.
\end{tablenotes}
\end{table}
"""

with open('Table_8_Robustness.tex', 'w', encoding='utf-8') as f:
    f.write(table8_latex)
print("  Table 8 저장: Table_8_Robustness.tex")


# =============================================================================
# 마스터 LaTeX 문서 생성 (모든 표 포함)
# =============================================================================
print("\n[11. 마스터 LaTeX 문서 생성]")

master_latex = r"""\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{threeparttable}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

\title{BACBB Strategy Performance Analysis: \\
Statistical Tables for Academic Publication}
\author{}
\date{}

\begin{document}

\maketitle

\section*{List of Tables}
\begin{itemize}
    \item Table 1: Descriptive Statistics of Daily Returns
    \item Table 2: Out-of-Sample Validation Results
    \item Table 3: Annual Performance by Year
    \item Table 4: Factor Regression Results
    \item Table 5: Conditional Performance Analysis
    \item Table 6: Rolling Window Sharpe Ratio Analysis
    \item Table 7: Comprehensive Performance Comparison
    \item Table 8: Robustness Analysis
\end{itemize}

\clearpage

% Table 1
\input{Table_1_Descriptive_Stats.tex}

\clearpage

% Table 2
\input{Table_2_OOS_Validation.tex}

\clearpage

% Table 3
\input{Table_3_Yearly_Performance.tex}

\clearpage

% Table 4
\input{Table_4_Regression.tex}

\clearpage

% Table 5
\input{Table_5_Conditional_Performance.tex}

\clearpage

% Table 6
\input{Table_6_Rolling_Analysis.tex}

\clearpage

% Table 7
\input{Table_7_Summary_Comparison.tex}

\clearpage

% Table 8
\input{Table_8_Robustness.tex}

\end{document}
"""

with open('BACBB_Tables_Master.tex', 'w', encoding='utf-8') as f:
    f.write(master_latex)
print("  마스터 문서 저장: BACBB_Tables_Master.tex")

# =============================================================================
# 최종 요약
# =============================================================================
print("\n" + "="*70)
print("STATA 스타일 LaTeX 표 생성 완료!")
print("="*70)
print(f"""
생성된 파일:
  1. Table_1_Descriptive_Stats.tex  - 기술통계량
  2. Table_2_OOS_Validation.tex     - Out-of-Sample 검증
  3. Table_3_Yearly_Performance.tex - 연도별 성과
  4. Table_4_Regression.tex         - 회귀분석 결과
  5. Table_5_Conditional_Performance.tex - 하락장 방어력
  6. Table_6_Rolling_Analysis.tex   - 롤링 윈도우 분석
  7. Table_7_Summary_Comparison.tex - 종합 성과 비교
  8. Table_8_Robustness.tex         - Robustness Check
  9. BACBB_Tables_Master.tex        - 마스터 문서

사용법:
  1. 개별 표: \\input{{Table_X_xxx.tex}} 로 논문에 삽입
  2. 전체 컴파일: pdflatex BACBB_Tables_Master.tex

필요한 LaTeX 패키지:
  - booktabs (전문적인 표 스타일)
  - threeparttable (표 주석)
  - multirow (다중 행)
  - amsmath (수학 기호)
""")

print("\n완료!")
