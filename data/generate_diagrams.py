"""
BACBB 논문 로직 구조도 및 펀딩비 구조 그림 생성
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("BACBB 논문 다이어그램 생성")
print("="*70)


# =============================================================================
# Figure 1: BACBB 전략 로직 구조도
# =============================================================================
print("\n[1] BACBB 전략 로직 구조도 생성 중...")

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# 색상 정의
colors = {
    'data': '#E3F2FD',
    'var': '#FFF3E0',
    'news': '#E8F5E9',
    'beta': '#FCE4EC',
    'portfolio': '#F3E5F5',
    'result': '#FFEB3B',
}

def draw_box(ax, x, y, w, h, text, color, fontsize=10, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                         facecolor=color, edgecolor='#424242', linewidth=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, wrap=True)

def draw_arrow(ax, start, end, color='#424242', style='->'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=2))

# 제목
ax.text(8, 11.5, 'BACBB (Betting Against Cryptocurrency Bad Beta) 전략 구조도',
        ha='center', va='center', fontsize=16, fontweight='bold')

# 1단계: 데이터 입력
ax.text(2, 10.3, '① 데이터 입력', fontsize=12, fontweight='bold', color='#1565C0')
draw_box(ax, 0.5, 9, 2.5, 0.8, '암호화폐\n가격/수익률', colors['data'], 9)
draw_box(ax, 3.5, 9, 2.5, 0.8, '거래량\n(시가총액 가중)', colors['data'], 9)
draw_box(ax, 6.5, 9, 2.5, 0.8, '국채 금리\n(Term Spread)', colors['data'], 9)
draw_box(ax, 9.5, 9, 2.5, 0.8, '펀딩비\n(Funding Rate)', colors['data'], 9)

# 2단계: VAR 모델
ax.text(2, 8.0, '② VAR(1) 모델 추정', fontsize=12, fontweight='bold', color='#E65100')
draw_box(ax, 1, 6.5, 10, 1.2, '', colors['var'], 9)
ax.text(6, 7.3, 'VAR(1) 상태변수 (z_t)', fontsize=11, fontweight='bold', ha='center')
ax.text(2.5, 6.8, 'z1: 시장 초과수익률', fontsize=9, ha='center')
ax.text(6, 6.8, 'z2: 기간 스프레드', fontsize=9, ha='center')
ax.text(9.5, 6.8, 'z3: 밸류에이션', fontsize=9, ha='center')


# VAR 수식
ax.text(13, 7.0, 'z(t+1) = c + A*z(t) + u(t+1)', fontsize=11, 
        ha='center', style='italic', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 화살표: 데이터 -> VAR
draw_arrow(ax, (1.75, 9), (2.5, 7.7))
draw_arrow(ax, (4.75, 9), (5, 7.7))
draw_arrow(ax, (7.75, 9), (7.5, 7.7))

# 3단계: Campbell-Shiller 분해
ax.text(2, 5.8, '③ Campbell-Shiller 분해', fontsize=12, fontweight='bold', color='#2E7D32')
draw_box(ax, 2, 4.3, 4, 1.2, 'Cash-Flow News (N_CF)\n영구적 현금흐름 충격', colors['news'], 9)
draw_box(ax, 8, 4.3, 4, 1.2, 'Discount Rate News (N_DR)\n일시적 할인율 변동', colors['news'], 9)

# 화살표: VAR -> News
draw_arrow(ax, (4, 6.5), (4, 5.5))
draw_arrow(ax, (8, 6.5), (10, 5.5))

# 4단계: Beta 추정
ax.text(2, 3.3, '④ Beta 추정', fontsize=12, fontweight='bold', color='#C2185B')
draw_box(ax, 2, 2, 4, 1, 'Cash-Flow Beta (β_CF)\n= Cov(r_i, N_CF) / Var(N_CF)', colors['beta'], 9)
draw_box(ax, 8, 2, 4, 1, 'FP Beta (β_FP)\n= ρ × (σ_i / σ_m)', colors['beta'], 9)

# 화살표: News -> Beta
draw_arrow(ax, (4, 4.3), (4, 3))
draw_arrow(ax, (10, 4.3), (10, 3))

# Bad Beta 강조
ax.text(4, 1.6, 'Bad Beta (나쁜 베타)', fontsize=9, ha='center', color='#C62828', fontweight='bold')
ax.text(10, 1.6, 'Total Beta', fontsize=9, ha='center', color='#2E7D32', fontweight='bold')

# 5단계: 포트폴리오 구성
ax.text(12.5, 5.8, '⑤ 포트폴리오 구성', fontsize=12, fontweight='bold', color='#7B1FA2')
draw_box(ax, 12.5, 4.3, 3, 1.2, 'Long (70%)\n저 CF Beta\n(현금흐름 방어)', colors['portfolio'], 9)
draw_box(ax, 12.5, 2.8, 3, 1.2, 'Short (30%)\n고 CF Beta\n(현금흐름 취약)', colors['portfolio'], 9)

# 화살표: Beta -> Portfolio
draw_arrow(ax, (6, 2.5), (12.5, 4.9))
draw_arrow(ax, (6, 2.5), (12.5, 3.4))

# 6단계: BACBB 수익률
ax.text(12.5, 1.8, '⑥ BACBB 수익률', fontsize=12, fontweight='bold', color='#F57F17')
draw_box(ax, 12.5, 0.3, 3, 1.2, 'BACBB 수익률\n샤프 1.04\nt-stat 2.79***', colors['result'], 10, bold=True)

# 화살표: Portfolio -> Result
draw_arrow(ax, (14, 2.8), (14, 1.5))

# 수식 박스
formula_box = FancyBboxPatch((0.5, 0.3), 11.5, 1.2, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor='#424242', linewidth=1.5)
ax.add_patch(formula_box)
ax.text(6.25, 1.1, 'BACBB 수익률 공식 (수식 13):', fontsize=10, fontweight='bold', ha='center')
ax.text(6.25, 0.6, 'r_BACBB = w_L * β_L^(-1) * (r_L - r_f - f) - w_S * β_H^(-1) * (r_H - r_f - f)',
        fontsize=10, ha='center', style='italic')

plt.tight_layout()
plt.savefig('Figure_BACBB_Logic_Flow.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  Figure_BACBB_Logic_Flow.png 저장 완료")


# =============================================================================
# Figure 2: 펀딩비 구조 그림
# =============================================================================
print("\n[2] 펀딩비 구조 그림 생성 중...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# 제목
ax.text(7, 9.5, '암호화폐 무기한 선물 펀딩비 (Funding Rate) 메커니즘',
        ha='center', va='center', fontsize=16, fontweight='bold')

# 상단: 가격 관계
ax.text(7, 8.7, '무기한 선물 가격 <-> 현물 가격 연동 메커니즘', 
        ha='center', fontsize=12, fontweight='bold', color='#1565C0')

# 현물 가격 박스
draw_box(ax, 1, 7.2, 3, 1, '현물 가격\n(Spot Price)', '#E3F2FD', 11, bold=True)

# 무기한 선물 가격 박스
draw_box(ax, 10, 7.2, 3, 1, '무기한 선물 가격\n(Perpetual Price)', '#FFF3E0', 11, bold=True)

# 양방향 화살표
ax.annotate('', xy=(9.8, 7.7), xytext=(4.2, 7.7),
            arrowprops=dict(arrowstyle='<->', color='#424242', lw=2.5))
ax.text(7, 7.9, '펀딩비로 가격 수렴 유도', ha='center', fontsize=10, fontweight='bold')

# 시나리오 1: 선물 > 현물 (양의 펀딩비)
scenario1_box = FancyBboxPatch((0.5, 4), 6, 2.8, boxstyle="round,pad=0.1",
                                facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2)
ax.add_patch(scenario1_box)

ax.text(3.5, 6.5, '시나리오 1: 선물 > 현물', ha='center', fontsize=11, fontweight='bold', color='#C62828')
ax.text(3.5, 6.0, '(양의 펀딩비, Positive Funding)', ha='center', fontsize=9, color='#C62828')

# Long -> Short 화살표
ax.text(1.5, 5.3, 'Long\n(매수자)', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFCDD2'))
ax.text(5.5, 5.3, 'Short\n(매도자)', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#C8E6C9'))
ax.annotate('', xy=(4.8, 5.3), xytext=(2.2, 5.3),
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=2))
ax.text(3.5, 5.7, '펀딩비 지급', ha='center', fontsize=9, color='#C62828')

ax.text(3.5, 4.5, '롱 포지션이 숏 포지션에게\n펀딩비를 지급', ha='center', fontsize=9)

# 시나리오 2: 선물 < 현물 (음의 펀딩비)
scenario2_box = FancyBboxPatch((7.5, 4), 6, 2.8, boxstyle="round,pad=0.1",
                                facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
ax.add_patch(scenario2_box)

ax.text(10.5, 6.5, '시나리오 2: 선물 < 현물', ha='center', fontsize=11, fontweight='bold', color='#2E7D32')
ax.text(10.5, 6.0, '(음의 펀딩비, Negative Funding)', ha='center', fontsize=9, color='#2E7D32')

# Short -> Long 화살표
ax.text(8.5, 5.3, 'Short\n(매도자)', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#C8E6C9'))
ax.text(12.5, 5.3, 'Long\n(매수자)', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFCDD2'))
ax.annotate('', xy=(11.8, 5.3), xytext=(9.2, 5.3),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2))
ax.text(10.5, 5.7, '펀딩비 지급', ha='center', fontsize=9, color='#2E7D32')

ax.text(10.5, 4.5, '숏 포지션이 롱 포지션에게\n펀딩비를 지급', ha='center', fontsize=9)


# 하단: 펀딩비 계산 및 정산
calc_box = FancyBboxPatch((0.5, 0.5), 13, 3.2, boxstyle="round,pad=0.1",
                           facecolor='#F5F5F5', edgecolor='#424242', linewidth=2)
ax.add_patch(calc_box)

ax.text(7, 3.4, '펀딩비 계산 및 정산', ha='center', fontsize=12, fontweight='bold')

# 계산 공식
ax.text(3.5, 2.7, '펀딩비 계산 공식:', fontsize=10, fontweight='bold', ha='center')
ax.text(3.5, 2.2, 'Funding Rate = Premium Index + clamp(Interest Rate - Premium Index)', 
        fontsize=9, ha='center', style='italic')
ax.text(3.5, 1.7, '* Premium Index: (선물가격 - 현물가격) / 현물가격', fontsize=9, ha='center')
ax.text(3.5, 1.3, '* Interest Rate: 기본 금리 (보통 0.01%)', fontsize=9, ha='center')

# 정산 주기
ax.text(10.5, 2.7, '정산 주기 (바이낸스):', fontsize=10, fontweight='bold', ha='center')
ax.text(10.5, 2.2, '* 8시간마다 정산 (00:00, 08:00, 16:00 UTC)', fontsize=9, ha='center')
ax.text(10.5, 1.7, '* 하루 3회, 연간 약 1,095회', fontsize=9, ha='center')
ax.text(10.5, 1.3, '* 평균 펀딩비: 약 0.01% (연 10.95%)', fontsize=9, ha='center')

# BACBB 관련 설명
ax.text(7, 0.8, 'BACBB 전략에서 펀딩비는 Long/Short 포지션의 비용으로 반영됨', 
        ha='center', fontsize=10, fontweight='bold', color='#1565C0')

plt.tight_layout()
plt.savefig('Figure_Funding_Rate_Structure.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  Figure_Funding_Rate_Structure.png 저장 완료")

# =============================================================================
# Figure 3: Campbell-Shiller 분해 상세 구조도
# =============================================================================
print("\n[3] Campbell-Shiller 분해 상세 구조도 생성 중...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# 제목
ax.text(7, 9.5, 'Campbell-Shiller 분해: 시장 수익률의 뉴스 분해',
        ha='center', va='center', fontsize=16, fontweight='bold')

# 상단: 시장 수익률
market_box = FancyBboxPatch((5, 8), 4, 1, boxstyle="round,pad=0.1",
                             facecolor='#BBDEFB', edgecolor='#1565C0', linewidth=2)
ax.add_patch(market_box)
ax.text(7, 8.5, '시장 수익률 (r_m)', ha='center', fontsize=12, fontweight='bold')

# 분해 화살표
ax.annotate('', xy=(4, 6.8), xytext=(6, 8),
            arrowprops=dict(arrowstyle='->', color='#424242', lw=2))
ax.annotate('', xy=(10, 6.8), xytext=(8, 8),
            arrowprops=dict(arrowstyle='->', color='#424242', lw=2))

ax.text(7, 7.5, 'Campbell-Shiller\n분해', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

# Cash-Flow News 박스
cf_box = FancyBboxPatch((1, 5), 5, 1.8, boxstyle="round,pad=0.1",
                         facecolor='#FFCDD2', edgecolor='#C62828', linewidth=2)
ax.add_patch(cf_box)
ax.text(3.5, 6.4, 'Cash-Flow News (N_CF)', ha='center', fontsize=11, fontweight='bold', color='#C62828')
ax.text(3.5, 5.9, '영구적 현금흐름 충격', ha='center', fontsize=10)
ax.text(3.5, 5.4, '* 펀더멘털에 대한 영구적 변화', ha='center', fontsize=9)

# Discount Rate News 박스
dr_box = FancyBboxPatch((8, 5), 5, 1.8, boxstyle="round,pad=0.1",
                         facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=2)
ax.add_patch(dr_box)
ax.text(10.5, 6.4, 'Discount Rate News (N_DR)', ha='center', fontsize=11, fontweight='bold', color='#2E7D32')
ax.text(10.5, 5.9, '일시적 할인율 변동', ha='center', fontsize=10)
ax.text(10.5, 5.4, '* 평균회귀하는 일시적 충격', ha='center', fontsize=9)


# 수식 박스
formula_box = FancyBboxPatch((1, 3), 12, 1.5, boxstyle="round,pad=0.1",
                              facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2)
ax.add_patch(formula_box)
ax.text(7, 4.2, '핵심 수식', ha='center', fontsize=11, fontweight='bold')
ax.text(7, 3.6, 'r(t+1) - E[r(t+1)] = N_CF(t+1) - N_DR(t+1)', 
        ha='center', fontsize=12, style='italic')

# Beta 분해
ax.text(7, 2.5, 'Beta 분해', ha='center', fontsize=12, fontweight='bold')

# CF Beta
cf_beta_box = FancyBboxPatch((1, 0.8), 5, 1.4, boxstyle="round,pad=0.1",
                              facecolor='#FFCDD2', edgecolor='#C62828', linewidth=2)
ax.add_patch(cf_beta_box)
ax.text(3.5, 1.9, 'Cash-Flow Beta (β_CF)', ha='center', fontsize=10, fontweight='bold', color='#C62828')
ax.text(3.5, 1.4, 'β_CF = Cov(r_i, N_CF) / Var(N_CF)', ha='center', fontsize=10)
ax.text(3.5, 0.95, '"Bad Beta" - 나쁜 베타', ha='center', fontsize=9, color='#C62828', fontweight='bold')

# DR Beta
dr_beta_box = FancyBboxPatch((8, 0.8), 5, 1.4, boxstyle="round,pad=0.1",
                              facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=2)
ax.add_patch(dr_beta_box)
ax.text(10.5, 1.9, 'Discount Rate Beta (β_DR)', ha='center', fontsize=10, fontweight='bold', color='#2E7D32')
ax.text(10.5, 1.4, 'β_DR = Cov(r_i, N_DR) / Var(N_DR)', ha='center', fontsize=10)
ax.text(10.5, 0.95, '"Good Beta" - 좋은 베타', ha='center', fontsize=9, color='#2E7D32', fontweight='bold')

# 화살표
ax.annotate('', xy=(3.5, 2.2), xytext=(3.5, 3),
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=2))
ax.annotate('', xy=(10.5, 2.2), xytext=(10.5, 3),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2))

plt.tight_layout()
plt.savefig('Figure_Campbell_Shiller_Decomposition.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  Figure_Campbell_Shiller_Decomposition.png 저장 완료")

print("\n" + "="*70)
print("다이어그램 생성 완료!")
print("="*70)
print("\n생성된 파일:")
print("  1. Figure_BACBB_Logic_Flow.png - BACBB 전략 로직 구조도")
print("  2. Figure_Funding_Rate_Structure.png - 펀딩비 구조 그림")
print("  3. Figure_Campbell_Shiller_Decomposition.png - Campbell-Shiller 분해 구조도")
