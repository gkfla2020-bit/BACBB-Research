# -*- coding: utf-8 -*-
"""
BACBB QuantStats HTML 리포트 생성
================================
VAR 모델 기반 Cash-Flow Beta 전략 성과 리포트
- BACBB vs BACB 비교
"""

import pandas as pd
import numpy as np
import quantstats as qs
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("BACBB QuantStats HTML 리포트 생성")
print("="*60)

# 수익률 데이터 로드
bacbb = pd.read_csv('bacbb_returns.csv', index_col=0, parse_dates=True)['BACBB']
bacb = pd.read_csv('bacb_returns.csv', index_col=0, parse_dates=True)['BACB']

# bh_returns.csv 존재 여부 확인
try:
    bh = pd.read_csv('bh_returns.csv', index_col=0, parse_dates=True)['BuyHold']
except:
    returns = pd.read_csv('06_daily_returns.csv', index_col=0, parse_dates=True)
    bh = returns.mean(axis=1)
    bh.name = 'BuyHold'

# 공통 인덱스
common_idx = bacbb.index.intersection(bacb.index).intersection(bh.index)
bacbb = bacbb.loc[common_idx]
bacb = bacb.loc[common_idx]
bh = bh.loc[common_idx]

print(f"\n분석 기간: {bacbb.index[0].strftime('%Y-%m-%d')} ~ {bacbb.index[-1].strftime('%Y-%m-%d')}")
print(f"거래일 수: {len(bacbb)}")

# QuantStats 설정
qs.extend_pandas()

# 1. BACBB vs BACB 리포트
print("\n[1. BACBB vs BACB 리포트 생성 중...]")
qs.reports.html(
    bacbb,
    benchmark=bacb,
    title='BACBB (Betting Against Cryptocurrency Bad Beta) vs BACB',
    output='BACBB_vs_BACB_QuantStats.html'
)
print("  BACBB_vs_BACB_QuantStats.html 생성 완료")

# 2. BACBB vs Buy & Hold 리포트
print("\n[2. BACBB vs Buy&Hold 리포트 생성 중...]")
qs.reports.html(
    bacbb,
    benchmark=bh,
    title='BACBB (Betting Against Cryptocurrency Bad Beta) vs Market',
    output='BACBB_vs_Market_QuantStats.html'
)
print("  BACBB_vs_Market_QuantStats.html 생성 완료")

# 3. BACB vs Buy & Hold 리포트
print("\n[3. BACB vs Buy&Hold 리포트 생성 중...]")
qs.reports.html(
    bacb,
    benchmark=bh,
    title='BACB (Betting Against Cryptocurrency Beta) vs Market',
    output='BACB_vs_Market_QuantStats.html'
)
print("  BACB_vs_Market_QuantStats.html 생성 완료")

# 콘솔 요약
print("\n" + "="*60)
print("성과 요약 비교")
print("="*60)

def print_stats(ret, name):
    print(f"\n[{name}]")
    print(f"  CAGR: {qs.stats.cagr(ret)*100:.2f}%")
    print(f"  Volatility: {qs.stats.volatility(ret)*100:.2f}%")
    print(f"  Sharpe: {qs.stats.sharpe(ret):.2f}")
    print(f"  Sortino: {qs.stats.sortino(ret):.2f}")
    print(f"  Max Drawdown: {qs.stats.max_drawdown(ret)*100:.2f}%")
    print(f"  Calmar: {qs.stats.calmar(ret):.2f}")
    print(f"  Win Rate: {qs.stats.win_rate(ret)*100:.1f}%")

print_stats(bacbb, "BACBB (Cash-Flow Beta)")
print_stats(bacb, "BACB (Total Beta)")
print_stats(bh, "Buy & Hold")

# 비교 테이블
print("\n" + "="*60)
print("비교 테이블")
print("="*60)
print(f"\n{'지표':<20} {'BACBB':>12} {'BACB':>12} {'B&H':>12}")
print("-"*60)
print(f"{'CAGR':<20} {qs.stats.cagr(bacbb)*100:>11.2f}% {qs.stats.cagr(bacb)*100:>11.2f}% {qs.stats.cagr(bh)*100:>11.2f}%")
print(f"{'Volatility':<20} {qs.stats.volatility(bacbb)*100:>11.2f}% {qs.stats.volatility(bacb)*100:>11.2f}% {qs.stats.volatility(bh)*100:>11.2f}%")
print(f"{'Sharpe':<20} {qs.stats.sharpe(bacbb):>12.2f} {qs.stats.sharpe(bacb):>12.2f} {qs.stats.sharpe(bh):>12.2f}")
print(f"{'Sortino':<20} {qs.stats.sortino(bacbb):>12.2f} {qs.stats.sortino(bacb):>12.2f} {qs.stats.sortino(bh):>12.2f}")
print(f"{'Max Drawdown':<20} {qs.stats.max_drawdown(bacbb)*100:>11.2f}% {qs.stats.max_drawdown(bacb)*100:>11.2f}% {qs.stats.max_drawdown(bh)*100:>11.2f}%")
print(f"{'Calmar':<20} {qs.stats.calmar(bacbb):>12.2f} {qs.stats.calmar(bacb):>12.2f} {qs.stats.calmar(bh):>12.2f}")
print(f"{'Win Rate':<20} {qs.stats.win_rate(bacbb)*100:>11.1f}% {qs.stats.win_rate(bacb)*100:>11.1f}% {qs.stats.win_rate(bh)*100:>11.1f}%")

print("\n" + "="*60)
print("QuantStats HTML 리포트 생성 완료!")
print("="*60)
print("\n생성된 파일:")
print("  1. BACBB_vs_BACB_QuantStats.html")
print("  2. BACBB_vs_Market_QuantStats.html")
print("  3. BACB_vs_Market_QuantStats.html")
