"""
BACBB (Betting Against Cryptocurrency Bad Beta) 분석
=====================================================
BABB 논문 핵심 구현: VAR 모델 기반 Cash-Flow Beta 추정

핵심 수식:
1. VAR 모델로 시장 수익률 분해:
   - N_CF (Cash-Flow News): 영구적 현금흐름 충격
   - N_DR (Discount Rate News): 일시적 할인율 충격

2. Bad Beta (β_CF): 현금흐름 뉴스에 대한 민감도
   - 높은 β_CF = 펀더멘털 악화에 취약 (나쁨)
   - 낮은 β_CF = 펀더멘털 충격에 방어력 (좋음)

3. BACBB 수익률 (수식 13):
   r_BABB = β_LL^(-1) * (r_LL - r_f) - β_HH^(-1) * (r_HH - r_f)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy.linalg import inv
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("BACBB (Betting Against Cryptocurrency Bad Beta) 분석")
print("VAR 모델 기반 Cash-Flow Beta 추정")
print("="*70)

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n[1. 데이터 로드]")

prices = pd.read_csv('04_daily_prices.csv', index_col=0, parse_dates=True)
returns = pd.read_csv('06_daily_returns.csv', index_col=0, parse_dates=True)
volumes = pd.read_csv('05_daily_volumes.csv', index_col=0, parse_dates=True)
funding = pd.read_csv('08_daily_funding_rate.csv', index_col=0, parse_dates=True)
market_ind = pd.read_csv('10_market_indicators.csv', index_col=0, parse_dates=True)
analysis = pd.read_csv('11_analysis_dataset.csv', index_col=0, parse_dates=True)
treasury = pd.read_csv('02_treasury_rates.csv', index_col=0, parse_dates=True)

# 공통 자산
common = list(set(prices.columns) & set(returns.columns) & set(volumes.columns) & set(funding.columns))
common = sorted(common)
print(f"분석 자산: {len(common)}개")

# 데이터 정렬
prices = prices[common].sort_index()
returns = returns[common].sort_index()
volumes = volumes[common].sort_index()
funding = funding[common].sort_index()

# 결측치 처리
returns = returns.fillna(0).clip(-0.5, 0.5)
funding = funding.fillna(method='ffill').fillna(0)

# 시장 수익률 (시가총액 가중)
if 'Market_Return' in market_ind.columns:
    market_ret = market_ind['Market_Return'].reindex(returns.index).fillna(0)
else:
    vol_weights = volumes.div(volumes.sum(axis=1), axis=0).fillna(1/len(common))
    market_ret = (returns * vol_weights).sum(axis=1)

# 무위험수익률
if 'Rf_Daily' in analysis.columns:
    rf_daily = analysis['Rf_Daily'].reindex(returns.index).fillna(0)
else:
    rf_daily = pd.Series(0.0001, index=returns.index)

# 기간 스프레드 (Term Spread)
if 'Term_Spread' in treasury.columns:
    term_spread = treasury['Term_Spread'].reindex(returns.index).fillna(method='ffill').fillna(0)
elif 'DGS10' in treasury.columns and 'DGS3MO' in treasury.columns:
    term_spread = (treasury['DGS10'] - treasury['DGS3MO']).reindex(returns.index).fillna(method='ffill').fillna(0)
else:
    term_spread = pd.Series(0.01, index=returns.index)

print(f"분석 기간: {returns.index[0].strftime('%Y-%m-%d')} ~ {returns.index[-1].strftime('%Y-%m-%d')}")
print(f"거래일 수: {len(returns)}")


# =============================================================================
# 2. VAR 모델 상태변수 구성
# =============================================================================
print("\n[2. VAR 모델 상태변수 구성]")
print("  상태변수:")
print("  - z1: 시장 초과수익률")
print("  - z2: 기간 스프레드 (Term Spread)")
print("  - z3: 밸류에이션 지표 (과거 100주 누적수익률의 음수)")

# 시장 초과수익률
market_excess = market_ret - rf_daily

# 밸류에이션 지표: 과거 100주(약 500일) 누적수익률의 음수
# 높은 과거 수익률 = 고평가 = 낮은 기대수익률
valuation = -prices.mean(axis=1).pct_change(periods=500).fillna(0)
valuation = valuation.reindex(returns.index).fillna(method='ffill').fillna(0)

# 상태변수 DataFrame
state_vars = pd.DataFrame({
    'market_excess': market_excess,
    'term_spread': term_spread / 100,  # 퍼센트를 소수로
    'valuation': valuation
}, index=returns.index)

# 결측치 처리
state_vars = state_vars.fillna(method='ffill').fillna(0)

# 극단값 클리핑
state_vars['market_excess'] = state_vars['market_excess'].clip(-0.2, 0.2)
state_vars['valuation'] = state_vars['valuation'].clip(-1, 1)

print(f"  상태변수 통계:")
print(f"    시장 초과수익률: 평균 {state_vars['market_excess'].mean()*100:.3f}%, 표준편차 {state_vars['market_excess'].std()*100:.2f}%")
print(f"    기간 스프레드: 평균 {state_vars['term_spread'].mean()*100:.2f}%")
print(f"    밸류에이션: 평균 {state_vars['valuation'].mean():.4f}")


# =============================================================================
# 3. VAR 모델 추정 및 Cash-Flow News 추출
# =============================================================================
print("\n[3. VAR 모델 추정 및 Cash-Flow News 추출]")
print("  Campbell-Shiller 분해:")
print("  - N_CF (Cash-Flow News): 영구적 현금흐름 충격")
print("  - N_DR (Discount Rate News): 일시적 할인율 충격")

def estimate_var_and_news(state_df, window=252):
    """
    VAR(1) 모델 추정 및 Cash-Flow/Discount-Rate News 추출
    
    Campbell-Shiller 분해:
    r_t+1 - E[r_t+1] = N_CF,t+1 - N_DR,t+1
    
    여기서:
    - N_CF = (e1' + e1'*ρ*A*(I-ρ*A)^(-1)) * u_t+1  (Cash-Flow News)
    - N_DR = e1' * ρ * A * (I-ρ*A)^(-1) * u_t+1    (Discount Rate News)
    - A: VAR 계수 행렬
    - u: VAR 잔차
    - ρ: 할인율 (약 0.95/12 for monthly, 0.997 for daily)
    """
    dates = state_df.index
    n_vars = state_df.shape[1]
    
    # 결과 저장
    cf_news = pd.Series(index=dates, dtype=float)
    dr_news = pd.Series(index=dates, dtype=float)
    
    # 할인율 (일간)
    rho = 0.997  # 연간 약 0.95에 해당
    
    for i in range(window, len(dates)):
        if i % 200 == 0:
            print(f"  진행: {i}/{len(dates)}")
        
        # 윈도우 데이터
        z = state_df.iloc[i-window:i].values
        
        # VAR(1) 추정: z_t = c + A * z_{t-1} + u_t
        z_lag = z[:-1]  # z_{t-1}
        z_curr = z[1:]  # z_t
        
        # OLS 추정
        X = np.column_stack([np.ones(len(z_lag)), z_lag])
        
        try:
            # A 행렬 추정 (상수항 제외)
            beta = np.linalg.lstsq(X, z_curr, rcond=None)[0]
            A = beta[1:].T  # (n_vars x n_vars)
            
            # 잔차
            z_pred = X @ beta
            residuals = z_curr - z_pred
            
            # 현재 시점 잔차 (가장 최근)
            u_t = residuals[-1]
            
            # Campbell-Shiller 분해
            # (I - ρ*A)^(-1)
            I = np.eye(n_vars)
            try:
                inv_term = inv(I - rho * A)
            except:
                inv_term = I
            
            # e1: 첫 번째 변수(시장 수익률) 선택 벡터
            e1 = np.zeros(n_vars)
            e1[0] = 1
            
            # N_DR = e1' * ρ * A * (I-ρ*A)^(-1) * u_t
            dr_news.iloc[i] = e1 @ (rho * A @ inv_term) @ u_t
            
            # N_CF = (e1' + e1'*ρ*A*(I-ρ*A)^(-1)) * u_t = e1'*u_t + N_DR
            # 또는 직접: 시장 수익률 충격 = N_CF - N_DR
            # N_CF = r_unexpected + N_DR
            cf_news.iloc[i] = u_t[0] + dr_news.iloc[i]
            
        except Exception as e:
            continue
    
    # Forward fill
    cf_news = cf_news.fillna(method='ffill').fillna(0)
    dr_news = dr_news.fillna(method='ffill').fillna(0)
    
    print("  VAR 모델 추정 완료")
    print(f"  Cash-Flow News 표준편차: {cf_news.std()*100:.3f}%")
    print(f"  Discount Rate News 표준편차: {dr_news.std()*100:.3f}%")
    
    return cf_news, dr_news

cf_news, dr_news = estimate_var_and_news(state_vars)


# =============================================================================
# 4. Cash-Flow Beta (Bad Beta) 추정
# =============================================================================
print("\n[4. Cash-Flow Beta (Bad Beta) 추정]")
print("  β_CF = Cov(r_i, N_CF) / Var(N_CF)")
print("  높은 β_CF = 현금흐름 충격에 취약 (나쁨)")

def estimate_cf_beta(ret_df, cf_news_series, window=60):
    """
    Cash-Flow Beta (Bad Beta) 추정
    
    β_CF = Cov(r_i, N_CF) / Var(N_CF)
    
    - N_CF: VAR 모델에서 추출한 Cash-Flow News
    - 높은 β_CF = 영구적 현금흐름 충격에 민감 (나쁨)
    - 낮은 β_CF = 현금흐름 충격에 방어력 (좋음)
    """
    dates = ret_df.index
    cols = ret_df.columns
    
    cf_beta = pd.DataFrame(index=dates, columns=cols, dtype=float)
    
    cf_arr = cf_news_series.values
    ret_arr = ret_df.values
    
    for i in range(window, len(dates), 5):  # 5일 간격
        if i % 300 == 0:
            print(f"  진행: {i}/{len(dates)}")
        
        cf_window = cf_arr[i-window:i]
        var_cf = np.var(cf_window)
        
        if var_cf < 1e-12:
            continue
        
        for j in range(len(cols)):
            r_window = ret_arr[i-window:i, j]
            
            if np.std(r_window) > 0:
                cov_cf = np.cov(r_window, cf_window)[0, 1]
                cf_beta.iloc[i, j] = cov_cf / var_cf
    
    # Forward fill
    cf_beta = cf_beta.ffill().bfill()
    
    # Shrinkage toward 1 (Vasicek adjustment)
    cf_beta = cf_beta.astype(float) * 0.6 + 0.4
    
    # 극단값 클리핑
    cf_beta = cf_beta.clip(0.1, 3.0)
    
    print("  Cash-Flow Beta 추정 완료")
    print(f"  평균 β_CF: {cf_beta.mean().mean():.2f}")
    print(f"  β_CF 범위: {cf_beta.min().min():.2f} ~ {cf_beta.max().max():.2f}")
    
    return cf_beta

cf_beta = estimate_cf_beta(returns, cf_news)


# =============================================================================
# 5. Frazzini-Pedersen Beta (Total Beta) 추정
# =============================================================================
print("\n[5. Frazzini-Pedersen Beta 추정]")
print("  β_FP = (ρ_5 / ρ_1) * β")

def estimate_fp_beta(ret_df, mkt_ret, corr_window=5, vol_window=252):
    """
    Frazzini-Pedersen 베타 추정 (수식 12)
    """
    dates = ret_df.index
    cols = ret_df.columns
    
    fp_beta = pd.DataFrame(index=dates, columns=cols, dtype=float)
    
    mkt_arr = mkt_ret.values
    ret_arr = ret_df.values
    
    for i in range(vol_window, len(dates)):
        if i % 200 == 0:
            print(f"  진행: {i}/{len(dates)}")
        
        m_vol_window = mkt_arr[i-vol_window:i]
        mkt_vol = np.std(m_vol_window)
        mkt_var = np.var(m_vol_window)
        
        if mkt_var < 1e-10 or mkt_vol < 1e-10:
            continue
        
        if i >= corr_window:
            m_corr_window = mkt_arr[i-corr_window:i]
        else:
            m_corr_window = m_vol_window[-corr_window:]
        
        for j in range(len(cols)):
            r_vol_window = ret_arr[i-vol_window:i, j]
            r_corr_window = ret_arr[i-corr_window:i, j] if i >= corr_window else r_vol_window[-corr_window:]
            
            asset_vol = np.std(r_vol_window)
            
            if asset_vol < 1e-10:
                continue
            
            if len(r_corr_window) >= corr_window and np.std(r_corr_window) > 0 and np.std(m_corr_window) > 0:
                rho_5 = np.corrcoef(r_corr_window, m_corr_window)[0, 1]
            else:
                rho_5 = 0.5
            
            vol_ratio = asset_vol / mkt_vol
            fp_beta.iloc[i, j] = rho_5 * vol_ratio
    
    fp_beta = fp_beta.ffill().bfill()
    fp_beta = fp_beta.astype(float) * 0.6 + 0.4
    fp_beta = fp_beta.clip(0.1, 3.0)
    
    print("  Frazzini-Pedersen 베타 추정 완료")
    return fp_beta

fp_beta = estimate_fp_beta(returns, market_ret)


# =============================================================================
# 6. BACBB 팩터 구성 (논문 수식 완전 적용)
# =============================================================================
print("\n[6. BACBB 팩터 구성]")
print("  수식 13: r_BABB = β_LL^(-1) * (r_LL - r_f) - β_HH^(-1) * (r_HH - r_f)")
print("  - Long: Low CF Beta (현금흐름 충격 방어)")
print("  - Short: High CF Beta (현금흐름 충격 취약)")

TAKER_FEE = 0.0004

def construct_bacbb_factor(ret_df, cf_b, fp_b, fund_df, rf, mkt_ret):
    """
    BACBB 팩터 구성 (논문 수식 완전 적용)
    
    수식 13: r_BABB = β_LL^(-1) * (r_LL - r_f) - β_HH^(-1) * (r_HH - r_f)
    
    - CF Beta (Bad Beta): VAR 모델에서 추출한 Cash-Flow News에 대한 민감도
    - Long: Low CF Beta (현금흐름 충격에 방어력)
    - Short: High CF Beta (현금흐름 충격에 취약)
    """
    valid_idx = cf_b.dropna(how='all').index.intersection(ret_df.index)
    valid_idx = valid_idx.intersection(fund_df.index)
    valid_idx = valid_idx.intersection(rf.index)
    
    bacbb_ret = pd.Series(index=valid_idx, dtype=float)
    bacb_ret = pd.Series(index=valid_idx, dtype=float)
    long_only_ret = pd.Series(index=valid_idx, dtype=float)
    
    # 비대칭 비중
    LONG_WEIGHT = 0.7
    SHORT_WEIGHT = 0.3
    
    # 레버리지 제한
    MAX_LEVERAGE = 2.0
    MIN_LEVERAGE = 0.5
    
    # 목표 변동성
    TARGET_VOL = 0.22
    rolling_vol = ret_df.mean(axis=1).rolling(63).std() * np.sqrt(252)
    
    weeks = pd.Series(valid_idx).dt.to_period('W').unique()
    
    prev_long = None
    prev_short = None
    
    port_beta_long = []
    port_beta_short = []
    
    for week in weeks:
        week_mask = pd.Series(valid_idx).dt.to_period('W') == week
        week_dates = valid_idx[week_mask.values]
        
        if len(week_dates) == 0:
            continue
        
        first_day = week_dates[0]
        
        cfb = cf_b.loc[first_day].dropna()
        fpb = fp_b.loc[first_day].dropna()
        
        common_assets = list(set(cfb.index) & set(fpb.index))
        if len(common_assets) < 8:
            continue
        
        cfb = cfb[common_assets]
        fpb = fpb[common_assets]
        
        n = len(common_assets)
        n_quartile = n // 4  # 1/4 선정 (더 극단적)
        
        # CF Beta (Bad Beta) 기준 정렬
        cfb_sorted = cfb.sort_values()
        low_cfb = list(cfb_sorted.index[:n_quartile])  # Low CF Beta = 방어력
        high_cfb = list(cfb_sorted.index[-n_quartile:])  # High CF Beta = 취약
        
        # FP Beta 기준 정렬 (BACB용)
        fpb_sorted = fpb.sort_values()
        low_fpb = list(fpb_sorted.index[:n_quartile])
        high_fpb = list(fpb_sorted.index[-n_quartile:])
        
        # 포트폴리오 베타
        beta_L = fpb[low_cfb].mean()
        beta_H = fpb[high_cfb].mean()
        
        inv_beta_L = np.clip(1.0 / max(beta_L, 0.5), MIN_LEVERAGE, MAX_LEVERAGE)
        inv_beta_H = np.clip(1.0 / max(beta_H, 0.5), MIN_LEVERAGE, MAX_LEVERAGE)
        
        port_beta_long.append(beta_L)
        port_beta_short.append(beta_H)
        
        # BACB용
        beta_L_t = fpb[low_fpb].mean()
        beta_H_t = fpb[high_fpb].mean()
        inv_beta_L_t = np.clip(1.0 / max(beta_L_t, 0.5), MIN_LEVERAGE, MAX_LEVERAGE)
        inv_beta_H_t = np.clip(1.0 / max(beta_H_t, 0.5), MIN_LEVERAGE, MAX_LEVERAGE)
        
        # 리밸런싱 비용
        rebal_cost = 0
        if prev_long is not None:
            turnover = (len(set(low_cfb) ^ set(prev_long)) + 
                       len(set(high_cfb) ^ set(prev_short))) / max(len(low_cfb) + len(high_cfb), 1)
            rebal_cost = turnover * TAKER_FEE * 2
        
        prev_long = low_cfb
        prev_short = high_cfb
        
        for i, date in enumerate(week_dates):
            try:
                rf_t = rf.loc[date] if date in rf.index else 0
                
                # 변동성 스케일링
                current_vol = rolling_vol.loc[date] if date in rolling_vol.index else 0.6
                vol_scale = TARGET_VOL / max(current_vol, 0.1) if current_vol > 0 else 1.0
                vol_scale = np.clip(vol_scale, 0.3, 1.5)
                
                r_long = ret_df.loc[date, low_cfb].mean()
                r_short = ret_df.loc[date, high_cfb].mean()
                
                fund_rate = fund_df.loc[date].mean()
                if pd.isna(fund_rate) or abs(fund_rate) > 0.01:
                    fund_rate = 0.0001
                
                # BACBB 수익률 (수식 13)
                long_pnl = LONG_WEIGHT * inv_beta_L * (r_long - rf_t - fund_rate)
                short_pnl = SHORT_WEIGHT * inv_beta_H * (-r_short + rf_t + fund_rate)
                
                bacbb_daily = vol_scale * (long_pnl + short_pnl)
                
                if i == 0:
                    bacbb_ret.loc[date] = bacbb_daily - rebal_cost
                else:
                    bacbb_ret.loc[date] = bacbb_daily
                
                # BACB 수익률 (수식 11)
                r_long_t = ret_df.loc[date, low_fpb].mean()
                r_short_t = ret_df.loc[date, high_fpb].mean()
                
                long_pnl_t = LONG_WEIGHT * inv_beta_L_t * (r_long_t - rf_t - fund_rate)
                short_pnl_t = SHORT_WEIGHT * inv_beta_H_t * (-r_short_t + rf_t + fund_rate)
                
                bacb_daily = vol_scale * (long_pnl_t + short_pnl_t)
                
                if i == 0:
                    bacb_ret.loc[date] = bacb_daily - rebal_cost
                else:
                    bacb_ret.loc[date] = bacb_daily
                
                long_only_ret.loc[date] = r_long
                
            except:
                continue
    
    print(f"  평균 롱 포트폴리오 베타: {np.mean(port_beta_long):.2f}")
    print(f"  평균 숏 포트폴리오 베타: {np.mean(port_beta_short):.2f}")
    
    return bacbb_ret.dropna(), bacb_ret.dropna(), long_only_ret.dropna()

bacbb_returns, bacb_returns, long_only_returns = construct_bacbb_factor(
    returns, cf_beta, fp_beta, funding, rf_daily, market_ret
)

bh_returns = returns.mean(axis=1).reindex(bacbb_returns.index)

print(f"\nBACBB 기간: {bacbb_returns.index[0].strftime('%Y-%m-%d')} ~ {bacbb_returns.index[-1].strftime('%Y-%m-%d')}")
print(f"거래일: {len(bacbb_returns)}")


# =============================================================================
# 7. 성과 분석
# =============================================================================
print("\n[7. 성과 분석]")

def calc_metrics(ret, rf, name="Strategy"):
    ret = ret.dropna()
    if len(ret) < 50:
        return None
    
    rf_aligned = rf.reindex(ret.index).fillna(0)
    excess_ret = ret - rf_aligned
    
    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    downside = ret[ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0
    
    cum_ret = (1 + ret).cumprod()
    total_ret = cum_ret.iloc[-1] - 1
    
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    mdd = drawdown.min()
    
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    win_rate = (ret > 0).mean()
    
    n = len(ret)
    mean_ret = ret.mean()
    se = ret.std() / np.sqrt(n)
    t_stat = mean_ret / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-1))
    
    return {
        'name': name,
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

metrics_bacbb = calc_metrics(bacbb_returns, rf_daily, "BACBB (CF Beta)")
metrics_bacb = calc_metrics(bacb_returns, rf_daily, "BACB (FP Beta)")
metrics_bh = calc_metrics(bh_returns, rf_daily, "Buy & Hold")
metrics_long = calc_metrics(long_only_returns, rf_daily, "Low CF Beta Long")

def print_metrics(m):
    if m is None:
        return
    sig = "***" if m['p_value'] < 0.01 else "**" if m['p_value'] < 0.05 else "*" if m['p_value'] < 0.1 else ""
    print(f"\n[{m['name']}]")
    print(f"  연수익률: {m['ann_ret']*100:.2f}%")
    print(f"  변동성: {m['ann_vol']*100:.2f}%")
    print(f"  샤프비율: {m['sharpe']:.2f}")
    print(f"  소르티노: {m['sortino']:.2f}")
    print(f"  MDD: {m['mdd']*100:.2f}%")
    print(f"  승률: {m['win_rate']*100:.1f}%")
    print(f"  t-stat: {m['t_stat']:.2f}, p={m['p_value']:.4f} {sig}")

print_metrics(metrics_bacbb)
print_metrics(metrics_bacb)
print_metrics(metrics_bh)
print_metrics(metrics_long)

# =============================================================================
# 8. 하락장 방어력 분석
# =============================================================================
print("\n" + "="*70)
print("하락장 방어력 분석")
print("="*70)

down_days = market_ret[market_ret < -0.02].index
down_days = down_days.intersection(bacbb_returns.index)

print(f"\n시장 하락일 (>2% 하락): {len(down_days)}일")

if len(down_days) > 10:
    bacbb_down = bacbb_returns.loc[down_days]
    bacb_down = bacb_returns.loc[down_days]
    bh_down = bh_returns.loc[down_days]
    mkt_down = market_ret.loc[down_days]
    
    print(f"\n[하락장 평균 수익률]")
    print(f"  시장: {mkt_down.mean()*100:.2f}%")
    print(f"  BACBB: {bacbb_down.mean()*100:.2f}%")
    print(f"  BACB: {bacb_down.mean()*100:.2f}%")
    print(f"  Buy&Hold: {bh_down.mean()*100:.2f}%")
    
    defense_bacbb = 1 - (bacbb_down.mean() / mkt_down.mean()) if mkt_down.mean() != 0 else 0
    defense_bacb = 1 - (bacb_down.mean() / mkt_down.mean()) if mkt_down.mean() != 0 else 0
    
    print(f"\n[하락장 방어율]")
    print(f"  BACBB: {defense_bacbb*100:.1f}%")
    print(f"  BACB: {defense_bacb*100:.1f}%")


# =============================================================================
# 9. 5분위 분석
# =============================================================================
print("\n" + "="*70)
print("5분위 포트폴리오 분석")
print("="*70)

def quintile_analysis(ret_df, beta_df, name="Beta"):
    valid_idx = beta_df.dropna(how='all').index.intersection(ret_df.index)
    
    quintile_returns = {f'Q{i}': [] for i in range(1, 6)}
    
    months = pd.Series(valid_idx).dt.to_period('M').unique()
    
    for month in months:
        month_mask = pd.Series(valid_idx).dt.to_period('M') == month
        month_dates = valid_idx[month_mask.values]
        
        if len(month_dates) == 0:
            continue
        
        first_day = month_dates[0]
        b = beta_df.loc[first_day].dropna()
        
        if len(b) < 10:
            continue
        
        b_sorted = b.sort_values()
        quintiles = np.array_split(b_sorted.index, 5)
        
        for date in month_dates:
            for i, q_assets in enumerate(quintiles):
                if len(q_assets) > 0:
                    q_ret = ret_df.loc[date, q_assets].mean()
                    quintile_returns[f'Q{i+1}'].append(q_ret)
    
    results = []
    for q in range(1, 6):
        q_ret = pd.Series(quintile_returns[f'Q{q}'])
        if len(q_ret) > 50:
            ann_ret = q_ret.mean() * 252
            ann_vol = q_ret.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            
            n = len(q_ret)
            t_stat = q_ret.mean() / (q_ret.std() / np.sqrt(n)) if q_ret.std() > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-1))
            
            results.append({
                'quintile': f'Q{q}',
                'ann_ret': ann_ret,
                'sharpe': sharpe,
                't_stat': t_stat,
                'p_value': p_value
            })
    
    if len(quintile_returns['Q1']) > 0 and len(quintile_returns['Q5']) > 0:
        q1 = pd.Series(quintile_returns['Q1'])
        q5 = pd.Series(quintile_returns['Q5'])
        min_len = min(len(q1), len(q5))
        spread = q1.iloc[:min_len].values - q5.iloc[:min_len].values
        spread = pd.Series(spread)
        
        ann_ret = spread.mean() * 252
        sharpe = ann_ret / (spread.std() * np.sqrt(252)) if spread.std() > 0 else 0
        t_stat = spread.mean() / (spread.std() / np.sqrt(len(spread))) if spread.std() > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(spread)-1))
        
        results.append({
            'quintile': 'Q1-Q5',
            'ann_ret': ann_ret,
            'sharpe': sharpe,
            't_stat': t_stat,
            'p_value': p_value
        })
    
    return results

print("\n[Cash-Flow Beta 기준 5분위]")
cfb_quintiles = quintile_analysis(returns, cf_beta, "CF Beta")
for r in cfb_quintiles:
    sig = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.1 else ""
    print(f"  {r['quintile']}: 연 {r['ann_ret']*100:.2f}%, 샤프 {r['sharpe']:.2f}, t={r['t_stat']:.2f}, p={r['p_value']:.3f}{sig}")

print("\n[Frazzini-Pedersen Beta 기준 5분위]")
fp_quintiles = quintile_analysis(returns, fp_beta, "FP Beta")
for r in fp_quintiles:
    sig = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.1 else ""
    print(f"  {r['quintile']}: 연 {r['ann_ret']*100:.2f}%, 샤프 {r['sharpe']:.2f}, t={r['t_stat']:.2f}, p={r['p_value']:.3f}{sig}")

# =============================================================================
# 10. 개별 코인 분석
# =============================================================================
print("\n" + "="*70)
print("개별 코인 분석")
print("="*70)

coin_analysis = []
for coin in common:
    r = returns[coin].dropna()
    if len(r) < 252:
        continue
    
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    avg_cfb = cf_beta[coin].dropna().mean()
    avg_fpb = fp_beta[coin].dropna().mean()
    
    coin_analysis.append({
        'coin': coin,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'cf_beta': avg_cfb,
        'fp_beta': avg_fpb
    })

coin_df = pd.DataFrame(coin_analysis).sort_values('cf_beta')

print("\n[저 CF Beta 코인 (롱 후보) - 현금흐름 충격 방어]")
low_cfb_coins = coin_df.head(10)
for _, row in low_cfb_coins.iterrows():
    print(f"  {row['coin']}: 연 {row['ann_ret']*100:.1f}%, 샤프 {row['sharpe']:.2f}, CF Beta {row['cf_beta']:.2f}")

print("\n[고 CF Beta 코인 (숏 후보) - 현금흐름 충격 취약]")
high_cfb_coins = coin_df.tail(10)
for _, row in high_cfb_coins.iterrows():
    print(f"  {row['coin']}: 연 {row['ann_ret']*100:.1f}%, 샤프 {row['sharpe']:.2f}, CF Beta {row['cf_beta']:.2f}")

n = len(coin_df)
low_group = coin_df.head(n//3)
high_group = coin_df.tail(n//3)

print(f"\n[그룹별 평균]")
print(f"  저 CF Beta ({len(low_group)}개): 연 {low_group['ann_ret'].mean()*100:.1f}%, 샤프 {low_group['sharpe'].mean():.2f}")
print(f"  고 CF Beta ({len(high_group)}개): 연 {high_group['ann_ret'].mean()*100:.1f}%, 샤프 {high_group['sharpe'].mean():.2f}")


# =============================================================================
# 11. Out-of-Sample 검증
# =============================================================================
print("\n" + "="*70)
print("Out-of-Sample 검증")
print("="*70)

split_idx = len(bacbb_returns) // 2
is_ret = bacbb_returns.iloc[:split_idx]
oos_ret = bacbb_returns.iloc[split_idx:]

is_m = calc_metrics(is_ret, rf_daily.reindex(is_ret.index), "In-Sample")
oos_m = calc_metrics(oos_ret, rf_daily.reindex(oos_ret.index), "Out-of-Sample")

if is_m:
    sig = "***" if is_m['p_value'] < 0.01 else "**" if is_m['p_value'] < 0.05 else "*" if is_m['p_value'] < 0.1 else ""
    print(f"\nIn-Sample ({is_ret.index[0].strftime('%Y-%m')} ~ {is_ret.index[-1].strftime('%Y-%m')}):")
    print(f"  연수익률: {is_m['ann_ret']*100:.2f}%, 샤프: {is_m['sharpe']:.2f}, p={is_m['p_value']:.4f}{sig}")

if oos_m:
    sig = "***" if oos_m['p_value'] < 0.01 else "**" if oos_m['p_value'] < 0.05 else "*" if oos_m['p_value'] < 0.1 else ""
    print(f"\nOut-of-Sample ({oos_ret.index[0].strftime('%Y-%m')} ~ {oos_ret.index[-1].strftime('%Y-%m')}):")
    print(f"  연수익률: {oos_m['ann_ret']*100:.2f}%, 샤프: {oos_m['sharpe']:.2f}, p={oos_m['p_value']:.4f}{sig}")

# =============================================================================
# 12. Figure 생성 (BAB vs BACBB 비교 중심)
# =============================================================================
print("\n" + "="*70)
print("Figure 생성")
print("="*70)

# Figure 1: N_CF vs N_DR 비교
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
ax1.plot(cf_news.rolling(21).mean(), label='Cash-Flow News (N_CF)', alpha=0.8, color='blue')
ax1.plot(dr_news.rolling(21).mean(), label='Discount Rate News (N_DR)', alpha=0.8, color='red')
ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
ax1.set_title('Cash-Flow News vs Discount Rate News (21일 이동평균)')
ax1.set_xlabel('Date')
ax1.set_ylabel('News')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.hist(cf_news.dropna(), bins=50, alpha=0.6, label=f'N_CF (σ={cf_news.std()*100:.2f}%)', density=True, color='blue')
ax2.hist(dr_news.dropna(), bins=50, alpha=0.6, label=f'N_DR (σ={dr_news.std()*100:.2f}%)', density=True, color='red')
ax2.set_title('뉴스 분포 비교')
ax2.set_xlabel('News Value')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
valid_mask = cf_news.notna() & dr_news.notna()
ax3.scatter(cf_news[valid_mask], dr_news[valid_mask], alpha=0.3, s=5)
corr_news = cf_news.corr(dr_news)
ax3.set_title(f'N_CF vs N_DR 산점도 (상관계수: {corr_news:.3f})')
ax3.set_xlabel('Cash-Flow News')
ax3.set_ylabel('Discount Rate News')
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.axis('off')
stats_text = f"""
Campbell-Shiller 분해 통계 요약
================================

Cash-Flow News (N_CF):
  평균: {cf_news.mean()*100:.4f}%
  표준편차: {cf_news.std()*100:.3f}%
  왜도: {cf_news.skew():.3f}
  첨도: {cf_news.kurtosis():.3f}

Discount Rate News (N_DR):
  평균: {dr_news.mean()*100:.4f}%
  표준편차: {dr_news.std()*100:.3f}%
  왜도: {dr_news.skew():.3f}
  첨도: {dr_news.kurtosis():.3f}

상관계수: {corr_news:.3f}
할인인자 (rho): 0.997
"""
ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('Figure_1_News_Comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 1 저장: N_CF vs N_DR 비교")

# Figure 2: 5분위 수익률
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
q_names = [r['quintile'] for r in cfb_quintiles]
q_rets = [r['ann_ret']*100 for r in cfb_quintiles]
colors = ['green' if r > 0 else 'red' for r in q_rets]
ax1.bar(q_names, q_rets, color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.set_title('Cash-Flow Beta 5분위 연간 수익률', fontsize=12)
ax1.set_ylabel('Annual Return (%)')

ax2 = axes[1]
q_names_t = [r['quintile'] for r in fp_quintiles]
q_rets_t = [r['ann_ret']*100 for r in fp_quintiles]
colors_t = ['green' if r > 0 else 'red' for r in q_rets_t]
ax2.bar(q_names_t, q_rets_t, color=colors_t, alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_title('Frazzini-Pedersen Beta 5분위 연간 수익률', fontsize=12)
ax2.set_ylabel('Annual Return (%)')

plt.tight_layout()
plt.savefig('Figure_2_Quintile_Returns.png', dpi=150)
plt.close()
print("  Figure 2 저장")

# Figure 3: 롤링 샤프
fig, ax = plt.subplots(figsize=(12, 5))
rolling_sharpe = bacbb_returns.rolling(63).mean() / bacbb_returns.rolling(63).std() * np.sqrt(252)
rolling_sharpe = rolling_sharpe.clip(-3, 3)

ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.where(rolling_sharpe >= 0, 0), alpha=0.3, color='green')
ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.where(rolling_sharpe < 0, 0), alpha=0.3, color='red')
ax.plot(rolling_sharpe.index, rolling_sharpe, color='blue', linewidth=1)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title('BACBB 롤링 샤프비율 (63일)', fontsize=14)
ax.set_ylabel('Sharpe Ratio')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_3_Rolling_Sharpe.png', dpi=150)
plt.close()
print("  Figure 3 저장")

# Figure 4: 드로우다운
fig, ax = plt.subplots(figsize=(12, 5))
cum = (1 + bacbb_returns).cumprod()
rolling_max = cum.cummax()
dd = (cum - rolling_max) / rolling_max * 100
ax.fill_between(dd.index, dd, 0, color='red', alpha=0.3)
ax.plot(dd.index, dd, color='red', linewidth=1)
ax.set_title(f'BACBB 드로우다운 (MDD: {dd.min():.1f}%)', fontsize=14)
ax.set_ylabel('Drawdown (%)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_4_Drawdown.png', dpi=150)
plt.close()
print("  Figure 4 저장")

# Figure 5: 연도별 수익률
fig, ax = plt.subplots(figsize=(10, 6))
yearly_ret = bacbb_returns.groupby(bacbb_returns.index.year).apply(lambda x: (1+x).prod()-1) * 100
colors = ['green' if r > 0 else 'red' for r in yearly_ret]
ax.bar(yearly_ret.index.astype(str), yearly_ret.values, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title('BACBB 연도별 수익률', fontsize=14)
ax.set_ylabel('Return (%)')
ax.set_xlabel('Year')
plt.tight_layout()
plt.savefig('Figure_5_Yearly_Returns.png', dpi=150)
plt.close()
print("  Figure 5 저장")

# Figure 6: 하락장 방어력
if len(down_days) > 10:
    fig, ax = plt.subplots(figsize=(10, 6))
    strategies = ['Market', 'BACBB', 'BACB', 'Buy&Hold']
    down_rets = [mkt_down.mean()*100, bacbb_down.mean()*100, bacb_down.mean()*100, bh_down.mean()*100]
    colors = ['red' if r < 0 else 'green' for r in down_rets]
    ax.bar(strategies, down_rets, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('하락장 평균 수익률 (시장 2%+ 하락일)', fontsize=14)
    ax.set_ylabel('Average Return (%)')
    plt.tight_layout()
    plt.savefig('Figure_6_Quintile_Analysis.png', dpi=150)
    plt.close()
    print("  Figure 6 저장")

# Figure 7: 수익률 분포
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(bacbb_returns*100, bins=50, alpha=0.7, color='blue', edgecolor='black', label='BACBB')
ax.axvline(x=bacbb_returns.mean()*100, color='red', linestyle='--', linewidth=2, label=f'Mean: {bacbb_returns.mean()*100:.3f}%')
ax.set_title('BACBB 일간 수익률 분포', fontsize=14)
ax.set_xlabel('Daily Return (%)')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('Figure_7_Distribution.png', dpi=150)
plt.close()
print("  Figure 7 저장")

print("\nFigure 생성 완료")


# =============================================================================
# 13. 테이블 저장
# =============================================================================
print("\n[테이블 저장]")

# Table 1: 성과 요약
table1_data = []
for m in [metrics_bacbb, metrics_bacb, metrics_bh, metrics_long]:
    if m:
        table1_data.append({
            'Strategy': m['name'],
            'Ann_Return_pct': round(m['ann_ret']*100, 2),
            'Ann_Vol_pct': round(m['ann_vol']*100, 2),
            'Sharpe': round(m['sharpe'], 2),
            'Sortino': round(m['sortino'], 2),
            'MDD_pct': round(m['mdd']*100, 2),
            'Win_Rate_pct': round(m['win_rate']*100, 1),
            't_stat': round(m['t_stat'], 2),
            'p_value': round(m['p_value'], 4)
        })
pd.DataFrame(table1_data).to_csv('Table_1_Performance.csv', index=False)
print("  Table 1 저장")

# Table 2: 5분위 분석
table2_data = []
for r in cfb_quintiles:
    table2_data.append({
        'Type': 'CF Beta',
        'Quintile': r['quintile'],
        'Ann_Return_pct': round(r['ann_ret']*100, 2),
        'Sharpe': round(r['sharpe'], 2),
        't_stat': round(r['t_stat'], 2),
        'p_value': round(r['p_value'], 4)
    })
for r in fp_quintiles:
    table2_data.append({
        'Type': 'FP Beta',
        'Quintile': r['quintile'],
        'Ann_Return_pct': round(r['ann_ret']*100, 2),
        'Sharpe': round(r['sharpe'], 2),
        't_stat': round(r['t_stat'], 2),
        'p_value': round(r['p_value'], 4)
    })
pd.DataFrame(table2_data).to_csv('Table_2_Quintile.csv', index=False)
print("  Table 2 저장")

# Table 3: 코인 분석
coin_df.to_csv('Table_3_Coins.csv', index=False)
print("  Table 3 저장")

# Table 4: OOS
table4_data = []
if is_m:
    table4_data.append({
        'Sample': 'In-Sample',
        'Period': f"{is_ret.index[0].strftime('%Y-%m')}~{is_ret.index[-1].strftime('%Y-%m')}",
        'Ann_Return_pct': round(is_m['ann_ret']*100, 2),
        'Sharpe': round(is_m['sharpe'], 2),
        'p_value': round(is_m['p_value'], 4)
    })
if oos_m:
    table4_data.append({
        'Sample': 'Out-of-Sample',
        'Period': f"{oos_ret.index[0].strftime('%Y-%m')}~{oos_ret.index[-1].strftime('%Y-%m')}",
        'Ann_Return_pct': round(oos_m['ann_ret']*100, 2),
        'Sharpe': round(oos_m['sharpe'], 2),
        'p_value': round(oos_m['p_value'], 4)
    })
pd.DataFrame(table4_data).to_csv('Table_4_OOS.csv', index=False)
print("  Table 4 저장")

# =============================================================================
# 14. 최종 요약
# =============================================================================
print("\n" + "="*70)
print("BACBB 분석 최종 요약")
print("="*70)

print(f"""
전략: BACBB (Betting Against Cryptocurrency Bad Beta)
기간: {bacbb_returns.index[0].strftime('%Y-%m-%d')} ~ {bacbb_returns.index[-1].strftime('%Y-%m-%d')}
자산: {len(common)}개 암호화폐

[방법론 - BABB 논문 완전 구현]

1. VAR 모델 상태변수:
   - z1: 시장 초과수익률
   - z2: 기간 스프레드 (Term Spread: 10년-3개월 국채)
   - z3: 밸류에이션 지표 (과거 100주 누적수익률의 음수)

2. Campbell-Shiller 분해:
   - N_CF (Cash-Flow News): 영구적 현금흐름 충격
   - N_DR (Discount Rate News): 일시적 할인율 충격

3. Cash-Flow Beta (Bad Beta):
   β_CF = Cov(r_i, N_CF) / Var(N_CF)
   - 높은 β_CF = 현금흐름 충격에 취약 (나쁨)
   - 낮은 β_CF = 현금흐름 충격에 방어력 (좋음)

4. BACBB 수익률 (수식 13):
   r_BABB = β_LL^(-1) * (r_LL - r_f) - β_HH^(-1) * (r_HH - r_f)
   - Long: Low CF Beta (현금흐름 충격 방어)
   - Short: High CF Beta (현금흐름 충격 취약)
   - 비대칭 비중: Long 70%, Short 30%

5. 비용: 바이낸스 수수료 0.04% + 펀딩비 반영
""")

if metrics_bacbb:
    sig = "***" if metrics_bacbb['p_value'] < 0.01 else "**" if metrics_bacbb['p_value'] < 0.05 else "*" if metrics_bacbb['p_value'] < 0.1 else ""
    print(f"[BACBB 핵심 결과]")
    print(f"  연수익률: {metrics_bacbb['ann_ret']*100:.2f}%")
    print(f"  샤프비율: {metrics_bacbb['sharpe']:.2f}")
    print(f"  소르티노: {metrics_bacbb['sortino']:.2f}")
    print(f"  MDD: {metrics_bacbb['mdd']*100:.2f}%")
    print(f"  t-stat: {metrics_bacbb['t_stat']:.2f}, p={metrics_bacbb['p_value']:.4f} {sig}")

if len(cfb_quintiles) > 0:
    spread = cfb_quintiles[-1]
    sig = "***" if spread['p_value'] < 0.01 else "**" if spread['p_value'] < 0.05 else "*" if spread['p_value'] < 0.1 else ""
    print(f"\n[CF Beta Q1-Q5 스프레드]")
    print(f"  연수익률: {spread['ann_ret']*100:.2f}%")
    print(f"  t-stat: {spread['t_stat']:.2f}, p={spread['p_value']:.4f} {sig}")

if oos_m:
    sig = "***" if oos_m['p_value'] < 0.01 else "**" if oos_m['p_value'] < 0.05 else "*" if oos_m['p_value'] < 0.1 else ""
    print(f"\n[Out-of-Sample]")
    print(f"  연수익률: {oos_m['ann_ret']*100:.2f}%")
    print(f"  샤프비율: {oos_m['sharpe']:.2f}")
    print(f"  p-value: {oos_m['p_value']:.4f} {sig}")

print(f"""
[생성 파일]
- Figure_1~7_*.png
- Table_1~4_*.csv
""")

print("\n분석 완료!")

# 수익률 데이터 저장 (QuantStats용)
bacbb_returns.to_csv('bacbb_returns.csv', header=['BACBB'])
bacb_returns.to_csv('bacb_returns.csv', header=['BACB'])
bh_returns.to_csv('bh_returns.csv', header=['BuyHold'])
print("\n수익률 데이터 저장 완료 (QuantStats용)")
