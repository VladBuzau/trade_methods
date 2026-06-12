"""
Genereaza STRATEGII.pdf — documentatie vizuala ChartVisualizer AutoTrader
Ruleaza: python generate_pdf.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Paleta de culori (dark theme) ─────────────────────────────────────────────
BG      = '#0d1117'
BG2     = '#161b22'
BG3     = '#21262d'
LIGHT   = '#e6edf3'
GRAY    = '#8b949e'
GRAY2   = '#484f58'
TEAL    = '#26a69a'
BLUE    = '#7986cb'
ORANGE  = '#ff9800'
YELLOW  = '#f0c040'
RED     = '#ef5350'
GREEN   = '#26a69a'
PURPLE  = '#9c27b0'
PINK    = '#e91e63'
CYAN    = '#00bcd4'
BULL    = '#26a69a'
BEAR    = '#ef5350'

# ── Font setup ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'text.color':      LIGHT,
    'axes.labelcolor': LIGHT,
    'xtick.color':     GRAY,
    'ytick.color':     GRAY,
    'axes.facecolor':  BG2,
    'figure.facecolor':BG,
    'axes.edgecolor':  GRAY2,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'grid.color':       GRAY2,
    'grid.linewidth':   0.4,
    'grid.alpha':       0.5,
})

# ── Helpers ───────────────────────────────────────────────────────────────────

def new_fig(w=11.7, h=8.3):
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    return fig

def ax_off(ax):
    ax.set_facecolor(BG)
    ax.axis('off')

def draw_candle(ax, x, o, c, h, l, w=0.55):
    col  = BULL if c >= o else BEAR
    yb   = min(o, c); ht = abs(c - o) or 0.0001
    rect = mpatches.Rectangle((x - w/2, yb), w, ht,
                               fc=col, ec=col, lw=0.5, zorder=3)
    ax.add_patch(rect)
    ax.plot([x, x], [l, yb],    color=col, lw=1.2, zorder=2)
    ax.plot([x, x], [yb+ht, h], color=col, lw=1.2, zorder=2)

def side_bar(ax, color=TEAL, w=0.012):
    rect = mpatches.Rectangle((0, 0), w, 1, fc=color, transform=ax.transAxes,
                               clip_on=False, zorder=10)
    ax.add_patch(rect)

def section_title(ax, text, sub='', color=TEAL, x=0.0, y=0.5):
    ax.set_facecolor(BG2)
    ax.axis('off')
    side_bar(ax, color)
    ax.text(0.025, 0.68, text, color=LIGHT, fontsize=13, fontweight='bold',
            transform=ax.transAxes, va='center')
    if sub:
        ax.text(0.025, 0.22, sub, color=GRAY, fontsize=8.5,
                transform=ax.transAxes, va='center')

def info_box(ax, lines, title='', bg=BG3, border=GRAY2, title_col=TEAL):
    ax.set_facecolor(bg)
    ax.axis('off')
    rect = mpatches.FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
        boxstyle='round,pad=0.01', fc=bg, ec=border, lw=1, transform=ax.transAxes)
    ax.add_patch(rect)
    y = 0.92
    if title:
        ax.text(0.06, y, title, color=title_col, fontsize=9, fontweight='bold',
                transform=ax.transAxes, va='top')
        y -= 0.12
    for line in lines:
        if isinstance(line, tuple):
            txt, col = line
        else:
            txt, col = line, LIGHT
        ax.text(0.06, y, txt, color=col, fontsize=8, transform=ax.transAxes,
                va='top', family='monospace')
        y -= 0.10
        if y < 0.05:
            break

def confidence_bar(ax, pct, label='', color=TEAL):
    ax.set_xlim(0, 100); ax.set_ylim(0, 1)
    ax.set_facecolor(BG3)
    ax.axis('off')
    ax.barh(0.35, 100, height=0.3, color=GRAY2, left=0)
    ax.barh(0.35, pct, height=0.3, color=color, left=0)
    ax.text(pct + 1, 0.35, f'{pct:.0f}%', color=LIGHT, fontsize=9,
            fontweight='bold', va='center')
    if label:
        ax.text(0, 0.82, label, color=GRAY, fontsize=7.5, va='center')

def badge(ax, text, x, y, color=TEAL, textcolor=BG, size=8):
    ax.text(x, y, f'  {text}  ', color=textcolor, fontsize=size,
            fontweight='bold', transform=ax.transAxes,
            bbox=dict(fc=color, ec=color, pad=2.5, boxstyle='round,pad=0.3'))

def gate_row(ax, i, text, passed=True, y_start=0.92, gap=0.11):
    y   = y_start - i * gap
    col = TEAL if passed else RED
    sym = '✓' if passed else '✗'
    ax.text(0.04, y, sym, color=col, fontsize=10, fontweight='bold',
            transform=ax.transAxes, va='top')
    ax.text(0.10, y, text, color=LIGHT if passed else GRAY,
            fontsize=8, transform=ax.transAxes, va='top')

def arrow_annotation(ax, x1, y1, x2, y2, text='', color=YELLOW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, text, color=color, fontsize=7.5, ha='center',
                bbox=dict(fc=BG2, ec=color, pad=2, boxstyle='round,pad=0.2'))

# ══════════════════════════════════════════════════════════════════════════════
#  PAGINI
# ══════════════════════════════════════════════════════════════════════════════

def page_cover(pdf):
    fig = new_fig()
    ax  = fig.add_axes([0, 0, 1, 1])
    ax_off(ax)

    # Gradient background effect cu linii
    for i in range(50):
        alpha = 0.015 + i * 0.004
        ax.axhline(i / 50, color=TEAL, lw=0.3, alpha=alpha)

    # Title block
    ax.text(0.5, 0.78, 'ChartVisualizer AutoTrader',
            color=LIGHT, fontsize=26, fontweight='bold', ha='center',
            transform=ax.transAxes)
    ax.text(0.5, 0.70, 'Ghid Complet Strategii',
            color=TEAL, fontsize=20, fontweight='bold', ha='center',
            transform=ax.transAxes)
    ax.text(0.5, 0.63, 'cu exemple de calcul si diagrame',
            color=GRAY, fontsize=12, ha='center', transform=ax.transAxes)

    # Separator line
    ax.axhline(0.60, xmin=0.1, xmax=0.9, color=TEAL, lw=1.5, alpha=0.7)

    # Strategy badges
    strategies = [
        ('EOB', PURPLE), ('SMC', BLUE), ('TrendRider', GREEN),
        ('SR_MTF', CYAN), ('Bollinger', CYAN), ('London Breakout', ORANGE),
        ('NY Breakout', ORANGE), ('RSI Divergence', PINK),
        ('Ichimoku', CYAN), ('Combined Mode', YELLOW), ('Scalp Boost', ORANGE),
    ]
    xs = np.linspace(0.05, 0.95, len(strategies))
    for (name, col), x in zip(strategies, xs):
        ax.text(x, 0.53, name, color=BG, fontsize=6.5, fontweight='bold',
                ha='center', transform=ax.transAxes,
                bbox=dict(fc=col, ec=col, pad=3, boxstyle='round,pad=0.3'))

    # Mini candlestick demo
    demo_ax = fig.add_axes([0.15, 0.15, 0.70, 0.30])
    demo_ax.set_facecolor(BG2)
    demo_ax.axis('off')

    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(30) * 0.5)
    for i in range(30):
        o = prices[i]
        c = prices[i] + np.random.randn() * 0.4
        h = max(o, c) + abs(np.random.randn() * 0.2)
        l = min(o, c) - abs(np.random.randn() * 0.2)
        draw_candle(demo_ax, i, o, c, h, l, w=0.55)

    # EOB zone
    z_lo, z_hi = 101.2, 102.0
    demo_ax.axhspan(z_lo, z_hi, alpha=0.25, color=TEAL, label='Order Block')
    demo_ax.text(30.5, (z_lo+z_hi)/2, 'Order Block ▶', color=TEAL,
                 fontsize=8, va='center')

    demo_ax.set_xlim(-1, 33)
    demo_ax.set_ylim(97, 104.5)
    demo_ax.text(0.5, -0.08, 'Exemplu vizual: detectie zona Order Block',
                 color=GRAY, fontsize=8, ha='center', transform=demo_ax.transAxes)

    ax.text(0.5, 0.07, f'Versiunea 3  |  {datetime.now().strftime("%B %Y")}',
            color=GRAY2, fontsize=9, ha='center', transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_toc(pdf):
    fig = new_fig()
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.1, top=0.95, bottom=0.04)
    ax_hdr = fig.add_subplot(gs[0])
    ax_body = fig.add_subplot(gs[1])

    section_title(ax_hdr, 'CUPRINS', color=TEAL)

    ax_body.set_facecolor(BG)
    ax_body.axis('off')

    toc = [
        ('1.  EOB + Unicorn',            'Zone Order Block pe H4 → BOS+FVG pe H1 → timing M5',            PURPLE),
        ('2.  SMC v3 STRICT',            '8 gate-uri: sweep → BOS → OB → FVG → displacement',             BLUE),
        ('3.  TrendRider',               '5 piloni: EMA + MACD + Supertrend + ADX + ROC',                  GREEN),
        ('4.  S/R Multi-Timeframe',      'Zone confirmate pe M15+H1+H4+D1 simultan',                       CYAN),
        ('5.  Bollinger Bands',          'Mean reversion: extremele benzilor + RSI',                       CYAN),
        ('6.  London Breakout',          'Range Asian 00-06 UTC → breakout la deschidere Londra',          ORANGE),
        ('7.  NY + China Session',       'Breakout NY 13:30 UTC + reversal sesiune Asiatica',              ORANGE),
        ('8.  RSI Divergence',           'Pret LL dar RSI HL → momentum epuizat → reversal',               PINK),
        ('9.  Engulfing + Ichimoku',     'Pattern price action + sistem japonez complet',                  PINK),
        ('10. Combined Mode',            'Vot majoritar ≥60% + spread ≥21% → executa',                    YELLOW),
        ('11. Scalp Boost Mode',         'Loturi ×2 + TP 1:1 + max 30min + N strat. agree',               ORANGE),
        ('12. Sistemul de Scoring',      'Cum se calculeaza confidence % si de ce 85% prag',               TEAL),
        ('13. Setari FTMO + FAQ',        'Configuratii sigure + intrebari frecvente',                      TEAL),
    ]

    y_start = 0.97
    for title, desc, col in toc:
        y = y_start - toc.index((title, desc, col)) * 0.073
        # Dot line
        ax_body.plot([0.03, 0.97], [y - 0.025, y - 0.025],
                     color=GRAY2, lw=0.5, linestyle=':', alpha=0.6,
                     transform=ax_body.transAxes)
        # Color dot
        ax_body.plot(0.03, y, 'o', color=col, markersize=6,
                     transform=ax_body.transAxes)
        ax_body.text(0.06, y, title, color=LIGHT, fontsize=9, fontweight='bold',
                     transform=ax_body.transAxes, va='center')
        ax_body.text(0.40, y, desc, color=GRAY, fontsize=8,
                     transform=ax_body.transAxes, va='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_eob(pdf):
    fig = new_fig()
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.45,
                            top=0.95, bottom=0.05, left=0.05, right=0.97)

    # Header
    ax_hdr = fig.add_subplot(gs[0, :])
    section_title(ax_hdr,
        '1. EOB + UNICORN — Enhanced Order Block',
        'Zone institutionale: H4 Order Block → MTF BOS+FVG → LTF timing',
        color=PURPLE)

    # ── Grafic 1: Order Block HTF ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1:3, 0])
    ax1.set_facecolor(BG2)
    ax1.set_title('HTF — Detectie Order Block', color=TEAL, fontsize=8.5, pad=5)

    # Candles: trend bearish → OB format → pret revine
    candles_htf = [
        # (o, c, h, l)
        (102.5, 103.2, 103.8, 102.2),  # 0 bullish
        (103.2, 104.1, 104.5, 103.0),  # 1 bullish
        (104.1, 103.5, 104.3, 103.2),  # 2 mic
        (103.5, 105.2, 105.5, 103.4),  # 3 BULLISH - OB candle
        (105.2, 103.0, 105.4, 102.8),  # 4 BEARISH MARE (engulf) → OB format
        (103.0, 102.2, 103.2, 101.9),  # 5 bear
        (102.2, 101.5, 102.4, 101.3),  # 6 bear
        (101.5, 100.8, 101.7, 100.6),  # 7 bear
        (100.8, 101.4, 101.6, 100.7),  # 8 retrasare spre OB
        (101.4, 102.8, 103.0, 101.3),  # 9 INTRARE in zona OB → BUY
        (102.8, 103.5, 103.7, 102.6),  # 10 bullish dupa OB
        (103.5, 104.2, 104.4, 103.4),  # 11 continuare
    ]
    for i, (o, c, h, l) in enumerate(candles_htf):
        draw_candle(ax1, i, o, c, h, l)

    # OB zona (lumanare 3 = OB candle)
    ob_lo, ob_hi = candles_htf[3][0], candles_htf[3][2]  # open si high
    ax1.axhspan(ob_lo, ob_hi, alpha=0.20, color=PURPLE, zorder=1)
    ax1.axhline(ob_lo, color=PURPLE, lw=1.2, linestyle='--', alpha=0.8, zorder=2)
    ax1.axhline(ob_hi, color=PURPLE, lw=1.2, linestyle='--', alpha=0.8, zorder=2)
    ax1.text(12.1, (ob_lo+ob_hi)/2, 'OB\nZona', color=PURPLE,
             fontsize=7.5, va='center', fontweight='bold')

    # Sageti adnotari
    ax1.annotate('OB Format\n(bearish engulf)', xy=(4, candles_htf[4][3]),
                 xytext=(5.5, 101.0),
                 arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.2),
                 color=PURPLE, fontsize=7, ha='center')
    ax1.annotate('Pret revine\nin zona OB', xy=(9, 101.4),
                 xytext=(7.5, 100.2),
                 arrowprops=dict(arrowstyle='->', color=YELLOW, lw=1.2),
                 color=YELLOW, fontsize=7, ha='center')
    ax1.text(9.3, 99.7, 'BUY ▶', color=GREEN, fontsize=8.5, fontweight='bold')

    ax1.set_xlim(-0.8, 14)
    ax1.set_ylim(99.5, 106)
    ax1.set_xlabel('Bare H4', fontsize=7.5, color=GRAY)
    ax1.set_ylabel('Pret', fontsize=7.5, color=GRAY)
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.3)

    # ── Grafic 2: BOS + FVG MTF ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1:3, 1])
    ax2.set_facecolor(BG2)
    ax2.set_title('MTF — BOS + FVG (H1)', color=TEAL, fontsize=8.5, pad=5)

    candles_mtf = [
        (100.8, 101.2, 101.5, 100.6),  # 0
        (101.2, 100.9, 101.4, 100.7),  # 1
        (100.9, 100.4, 101.0, 100.2),  # 2 LL
        (100.4, 100.1, 100.6, 99.9),   # 3 LL2 - FVG GAP
        (100.1, 101.0, 101.2, 100.0),  # 4 — FVG gap (bara 3 hi = 100.6, bara 5 lo = 100.8?)
        (101.0, 101.8, 102.0, 100.9),  # 5 BOS candle (depaseste swing high 101.5)
        (101.8, 102.3, 102.5, 101.6),  # 6 continuare
        (102.3, 101.9, 102.6, 101.8),  # 7
        (101.9, 102.6, 102.8, 101.7),  # 8
    ]
    for i, (o, c, h, l) in enumerate(candles_mtf):
        draw_candle(ax2, i, o, c, h, l)

    # Swing high level
    swing_high = 101.5
    ax2.axhline(swing_high, color=YELLOW, lw=1.2, linestyle=':', alpha=0.9, xmax=0.72)
    ax2.text(-0.5, swing_high + 0.04, f'Swing High {swing_high}', color=YELLOW,
             fontsize=7, va='bottom')

    # BOS annotation
    ax2.annotate(f'BOS!\nclose>{swing_high}', xy=(5, 101.8),
                 xytext=(3.5, 102.4),
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.3),
                 color=GREEN, fontsize=7.5, fontweight='bold', ha='center')

    # FVG zona (gap intre bara 3 high si bara 5 low)
    fvg_lo, fvg_hi = 100.6, 100.9  # gap fictiv vizibil
    ax2.axhspan(fvg_lo, fvg_hi, alpha=0.25, color=BLUE, zorder=1)
    ax2.text(9.0, (fvg_lo+fvg_hi)/2, 'FVG', color=BLUE,
             fontsize=7.5, va='center', fontweight='bold')

    ax2.set_xlim(-0.8, 10)
    ax2.set_ylim(99.6, 103.0)
    ax2.set_xlabel('Bare H1', fontsize=7.5, color=GRAY)
    ax2.tick_params(labelsize=7)
    ax2.grid(True, alpha=0.3)

    # ── Gate-uri panel ────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor(BG3)
    ax3.axis('off')
    ax3.text(0.5, 0.97, 'GATE-URI OBLIGATORII', color=TEAL, fontsize=8,
             fontweight='bold', ha='center', transform=ax3.transAxes)
    gates = [
        ('G1  Sesiune London/NY activa', True),
        ('G2  HTF trend + aliniat cu semnalul', True),
        ('G3  Zona OB cu pret in interior (5%)', True),
        ('G4  MTF BOS Grade A sau B', True),
        ('G5  Entry Score >= 6/10 pe LTF', True),
    ]
    for i, (g, p) in enumerate(gates):
        gate_row(ax3, i, g, p, y_start=0.85, gap=0.15)

    # ── Calcul Confidence ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 2])
    ax4.set_facecolor(BG3)
    ax4.axis('off')
    ax4.text(0.5, 0.97, 'CALCUL CONFIDENCE', color=PURPLE, fontsize=8,
             fontweight='bold', ha='center', transform=ax4.transAxes)

    calc_lines = [
        ('MAX SCORE = 20 puncte', YELLOW),
        ('', GRAY),
        ('HTF:  decay(3-5) + sweet(+2)', GRAY),
        ('      + vol(+2) + stack(+2) = 10', GRAY),
        ('MTF:  BOS(1-3) + FVG(+2)', GRAY),
        ('      + conf(+2) + sweep(+1) = 6', GRAY),
        ('LTF:  DNA(4) + disp(+3) = 6', GRAY),
        ('', GRAY),
        ('Ex: 17/20 = 85% ✓', GREEN),
        ('    14/20 = 70% → HOLD', RED),
    ]
    y = 0.87
    for txt, col in calc_lines:
        ax4.text(0.06, y, txt, color=col, fontsize=7.5, transform=ax4.transAxes,
                 va='top', family='monospace')
        y -= 0.088

    # ── Confidence bars ───────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    ax5.set_facecolor(BG)
    ax5.axis('off')

    examples = [
        ('HTF decay 0.8 + sweet spot', 4+2, 10, PURPLE),
        ('MTF BOS Grade A + FVG overlap 78%', 3+2+2, 6, BLUE),
        ('LTF compresie + displacement', 2+3, 6, GREEN),
    ]
    for j, (label, score, maxs, col) in enumerate(examples):
        pct = score / maxs * 100
        bax = fig.add_axes([0.05 + j*0.32, 0.05, 0.28, 0.035])
        bax.set_facecolor(BG3)
        bax.set_xlim(0, maxs); bax.set_ylim(0, 1)
        bax.axis('off')
        bax.barh(0.3, maxs, height=0.4, color=GRAY2)
        bax.barh(0.3, score, height=0.4, color=col)
        bax.text(score + 0.1, 0.3, f'{score}/{maxs} = {pct:.0f}%',
                 color=LIGHT, fontsize=8, va='center', fontweight='bold')
        bax.text(0, -0.4, label, color=GRAY, fontsize=7, va='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_smc(pdf):
    fig = new_fig()
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.45,
                            top=0.95, bottom=0.05, left=0.05, right=0.97)

    ax_hdr = fig.add_subplot(gs[0, :])
    section_title(ax_hdr,
        '2. SMC v3 STRICT — Smart Money Concepts',
        'Secventa completa: Sweep → BOS → OB → FVG → Displacement (8 gate-uri)',
        color=BLUE)

    # ── Diagrama structura piata ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1:, 0:2])
    ax1.set_facecolor(BG2)
    ax1.set_title('Anatomia unui setup SMC BEAR (GBPUSD H1)', color=TEAL, fontsize=8.5, pad=5)

    bear_candles = [
        # Setup: HH → LH (market structure BEAR)
        (100.0, 101.2, 101.5, 99.8),   # 0 bull
        (101.2, 101.8, 102.2, 101.0),  # 1 HH
        (101.8, 101.2, 102.0, 101.0),  # 2 pullback
        (101.2, 101.5, 101.7, 101.0),  # 3 LH (mai jos ca HH)
        (101.5, 100.5, 101.6, 100.3),  # 4 OB BEARISH (lumanare bullish urmata de sell)
        (100.5, 102.0, 102.4, 100.4),  # 5 LIQUIDITY SWEEP (spike sus, vaneaza stops)
        (102.0, 100.2, 102.1, 100.0),  # 6 BOS + displacement (corp mare, close<LL)
        (100.2, 100.6, 100.8, 100.1),  # 7 retrasare spre OB
        (100.6, 101.3, 101.4, 100.5),  # 8 pret in OB
        (101.3, 100.8, 101.5, 100.6),  # 9 confirmare in OB
        (100.8, 100.1, 101.0, 99.9),   # 10 SELL executat
        (100.1, 99.5, 100.3, 99.4),    # 11 miscare
        (99.5, 99.0, 99.7, 98.9),      # 12 TP atins
    ]
    for i, (o, c, h, l) in enumerate(bear_candles):
        draw_candle(ax1, i, o, c, h, l)

    # Structura (swing highs/lows)
    # HH la bara 1, LH la bara 3
    ax1.plot([1], [102.2], '^', color=YELLOW, ms=8, zorder=5)
    ax1.text(1, 102.35, 'HH', color=YELLOW, fontsize=7.5, ha='center', fontweight='bold')
    ax1.plot([3], [101.7], '^', color=ORANGE, ms=8, zorder=5)
    ax1.text(3, 101.85, 'LH', color=ORANGE, fontsize=7.5, ha='center', fontweight='bold')
    ax1.plot([2], [101.0], 'v', color=YELLOW, ms=8, zorder=5)
    ax1.text(2, 100.75, 'HL', color=YELLOW, fontsize=7.5, ha='center', fontweight='bold')

    # Swing low level (pentru BOS)
    sl_level = 100.3
    ax1.axhline(sl_level, color=RED, lw=1.2, linestyle=':', alpha=0.8, xmin=0.3, xmax=0.55)
    ax1.text(-0.4, sl_level - 0.08, f'Swing Low\n{sl_level}', color=RED,
             fontsize=6.5, va='top')

    # OB zona (bara 4)
    ob_lo = bear_candles[4][0]  # open
    ob_hi = bear_candles[4][2]  # high
    ax1.axhspan(ob_lo, ob_hi, alpha=0.22, color=RED, zorder=1)
    ax1.text(13.2, (ob_lo+ob_hi)/2, 'OB\nBEAR', color=RED,
             fontsize=7.5, va='center', fontweight='bold')

    # Liquidity sweep annotation
    ax1.annotate('SWEEP!\nVaneaza stops BUY', xy=(5, 102.4),
                 xytext=(3.5, 103.1),
                 arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.3),
                 color=ORANGE, fontsize=7, ha='center', fontweight='bold')

    # BOS annotation
    ax1.annotate(f'BOS!\nclose<{sl_level}\ncorp 82%', xy=(6, 100.2),
                 xytext=(7.5, 99.3),
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.3),
                 color=RED, fontsize=7, ha='center', fontweight='bold')

    # FVG zona (bara 6-8 gap)
    fvg_lo, fvg_hi = 100.2, 100.5
    ax1.axhspan(fvg_lo, fvg_hi, alpha=0.22, color=BLUE, zorder=1)
    ax1.text(13.2, (fvg_lo+fvg_hi)/2, 'FVG', color=BLUE,
             fontsize=7.5, va='center', fontweight='bold')

    # SELL entry + SL/TP
    sell_price = 101.0
    sl_price   = 102.5
    tp_price   = 99.0
    ax1.axhline(sell_price, color=RED,    lw=1.5, linestyle='--', alpha=0.9, xmin=0.55)
    ax1.axhline(sl_price,   color=ORANGE, lw=1.0, linestyle=':', alpha=0.8, xmin=0.55)
    ax1.axhline(tp_price,   color=GREEN,  lw=1.0, linestyle=':', alpha=0.8, xmin=0.55)
    ax1.text(13.2, sell_price, 'SELL', color=RED,    fontsize=7.5, va='center', fontweight='bold')
    ax1.text(13.2, sl_price,   'SL',   color=ORANGE, fontsize=7.5, va='center')
    ax1.text(13.2, tp_price,   'TP',   color=GREEN,  fontsize=7.5, va='center')

    ax1.set_xlim(-0.8, 14.5)
    ax1.set_ylim(98.6, 103.5)
    ax1.set_xlabel('Bare H1', fontsize=7.5, color=GRAY)
    ax1.set_ylabel('Pret', fontsize=7.5, color=GRAY)
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.3)

    # ── 8 Gate-uri panel ─────────────────────────────────────────────────────
    ax_gates = fig.add_subplot(gs[1, 2])
    ax_gates.set_facecolor(BG3)
    ax_gates.axis('off')
    ax_gates.text(0.5, 0.98, '8 GATE-URI OBLIGATORII', color=BLUE, fontsize=8,
                  fontweight='bold', ha='center', transform=ax_gates.transAxes)
    gates = [
        'G1  Market structure BULL/BEAR (nu RANGE)',
        'G2  BOS in 20 bare + corp >= 55%',
        'G3  OB fresh untested (max 50 bare)',
        'G4  FVG activ in zona OB',
        'G5  Liquidity sweep inainte de BOS',
        'G6  ATR 0.5x-2.0x media (normal)',
        'G7  Pret in 15% distanta de OB',
        'G8  2/3 TF-uri conviction >= 7/12',
    ]
    for i, g in enumerate(gates):
        gate_row(ax_gates, i, g, True, y_start=0.87, gap=0.10)

    # ── Conviction scale ─────────────────────────────────────────────────────
    ax_conv = fig.add_subplot(gs[2, 2])
    ax_conv.set_facecolor(BG3)
    ax_conv.axis('off')
    ax_conv.text(0.5, 0.97, 'CONVICTION / MAX 12', color=BLUE, fontsize=8,
                 fontweight='bold', ha='center', transform=ax_conv.transAxes)

    labels = ['BOS Grade A\n+3', 'OB in range\n+3', 'FVG overlap\n+2',
              'Sweep\n+1', 'Volume\n+2', 'Fresh OB\n+1']
    vals   = [3, 3, 2, 1, 2, 1]
    colors = [GREEN, TEAL, BLUE, ORANGE, PURPLE, CYAN]
    bars = ax_conv.barh(range(len(vals)), vals, color=colors, height=0.5,
                        left=0, align='center')
    ax_conv.set_xlim(0, 5)
    for j, (v, lbl) in enumerate(zip(vals, labels)):
        ax_conv.text(v + 0.1, j, f'+{v}', color=LIGHT, fontsize=8, va='center',
                     fontweight='bold')
    ax_conv.set_yticks(range(len(labels)))
    ax_conv.set_yticklabels(labels, fontsize=6.5)
    ax_conv.tick_params(labelsize=7)
    ax_conv.set_xlabel('Puncte', fontsize=7.5, color=GRAY)
    ax_conv.text(0, -1.2, 'Prag semnal: >= 7/12 per TF', color=YELLOW,
                 fontsize=7.5, fontweight='bold')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_trend_rider(pdf):
    fig = new_fig()
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.45,
                            top=0.95, bottom=0.06, left=0.07, right=0.97)

    ax_hdr = fig.add_subplot(gs[0, :])
    section_title(ax_hdr,
        '3. TRENDRIDER — Trend Following Unificat',
        '5 piloni: EMA Alignment + MACD + Supertrend + ADX + ROC Momentum',
        color=GREEN)

    # ── Grafic principal: EMA + Supertrend + price ────────────────────────────
    ax1 = fig.add_subplot(gs[1:, 0:2])
    ax1.set_facecolor(BG2)
    ax1.set_title('USDJPY H1 — Exemplu BUY: toti 5 piloni aliniati', color=TEAL, fontsize=8.5, pad=5)

    np.random.seed(7)
    n = 60
    base = np.linspace(149.5, 153.8, n) + np.random.randn(n) * 0.18
    emas8  = np.convolve(base, np.ones(8)/8, mode='same')
    emas21 = np.convolve(base, np.ones(21)/21, mode='same')
    emas50 = np.convolve(base, np.ones(50)/50, mode='same')
    xs = np.arange(n)

    # Lumânari simplificate
    for i in range(0, n, 2):
        o = base[i]
        c = base[i] + np.random.randn() * 0.15
        h = max(o, c) + 0.08
        l = min(o, c) - 0.08
        draw_candle(ax1, i, o, c, h, l, w=1.2)

    ax1.plot(xs, emas8,  color=GREEN,  lw=1.5, label='EMA 8',  alpha=0.9)
    ax1.plot(xs, emas21, color=YELLOW, lw=1.5, label='EMA 21', alpha=0.9)
    ax1.plot(xs, emas50, color=RED,    lw=1.5, label='EMA 50', alpha=0.9)

    # Supertrend (sub pret = bull)
    st_line = base - 0.45 - np.random.randn(n) * 0.05
    ax1.plot(xs, st_line, color=TEAL, lw=2, linestyle='--',
             label='Supertrend', alpha=0.8)
    ax1.fill_between(xs, st_line, base, alpha=0.07, color=GREEN)

    # Semnal BUY la bara 45
    ax1.annotate('BUY\nToti 5 piloni ✓', xy=(45, base[45]),
                 xytext=(35, 151.8),
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
                 color=GREEN, fontsize=8, fontweight='bold', ha='center',
                 bbox=dict(fc=BG3, ec=GREEN, pad=3, boxstyle='round,pad=0.3'))

    ax1.legend(fontsize=7, loc='upper left', framealpha=0.6,
               facecolor=BG3, edgecolor=GRAY2)
    ax1.set_xlabel('Bare H1', fontsize=7.5, color=GRAY)
    ax1.set_ylabel('USDJPY', fontsize=7.5, color=GRAY)
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, n+1)

    # ── 5 Piloni scoring table ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 2])
    ax2.set_facecolor(BG3)
    ax2.axis('off')
    ax2.text(0.5, 0.98, 'SCORING — 5 PILONI', color=GREEN, fontsize=8.5,
             fontweight='bold', ha='center', transform=ax2.transAxes)

    piloni = [
        ('EMA 8>21>50 aliniat',     '+2', GREEN),
        ('MACD line>sig + hist ↑',  '+2', GREEN),
        ('Supertrend sub pret',      '+2', GREEN),
        ('ADX > 25 (gate)',          '✓',  TEAL),
        ('ROC > 0.15% / 10 bare',   '+1', GREEN),
        ('─────────────────', '──', GRAY2),
        ('Scor bull total',          '7/9', YELLOW),
        ('Conv + bonus ADX',         '9',   YELLOW),
        ('Semnal la >= 5 pt',        'BUY', GREEN),
    ]
    y = 0.87
    for label, val, col in piloni:
        ax2.text(0.06, y, label, color=LIGHT if col != GRAY2 else GRAY2,
                 fontsize=7.5, transform=ax2.transAxes, va='top', family='monospace')
        ax2.text(0.85, y, val,   color=col,
                 fontsize=8, transform=ax2.transAxes, va='top',
                 fontweight='bold', ha='right')
        y -= 0.094

    # ── SL/TP calcul ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 2])
    ax3.set_facecolor(BG3)
    ax3.axis('off')
    ax3.text(0.5, 0.98, 'CALCUL SL / TP', color=GREEN, fontsize=8.5,
             fontweight='bold', ha='center', transform=ax3.transAxes)

    sl_tp_lines = [
        ('Pret:  153.40', LIGHT),
        ('ATR:     0.35', LIGHT),
        ('Supertrend: 152.65', LIGHT),
        ('', GRAY),
        ('SL = min(ST, pret-1.5×ATR)', GRAY),
        ('   = min(152.65, 152.875)', GRAY),
        ('   = 152.65', ORANGE),
        ('', GRAY),
        ('TP = pret + (pret-SL) × 1.5', GRAY),
        ('   = 153.40 + 0.75 × 1.5', GRAY),
        ('   = 154.525', GREEN),
        ('', GRAY),
        ('RR = 1 : 1.5', YELLOW),
    ]
    y = 0.87
    for txt, col in sl_tp_lines:
        ax3.text(0.06, y, txt, color=col, fontsize=7.5, transform=ax3.transAxes,
                 va='top', family='monospace')
        y -= 0.065

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_srmtf(pdf):
    fig = new_fig()
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.45,
                            top=0.95, bottom=0.06, left=0.07, right=0.97)

    ax_hdr = fig.add_subplot(gs[0, :])
    section_title(ax_hdr,
        '4. S/R MULTI-TIMEFRAME — Zone Confirmate Multi-TF',
        'Suport/Rezistenta valid DOAR daca apare pe cel putin 2 TF-uri simultan',
        color=CYAN)

    # ── Grafic principal: 4 TF-uri overlapped ────────────────────────────────
    ax1 = fig.add_subplot(gs[1:, 0:2])
    ax1.set_facecolor(BG2)
    ax1.set_title('XAUUSD — Zone S/R pe 4 TF-uri (M15 + H1 + H4 + D1)', color=TEAL, fontsize=8.5, pad=5)

    np.random.seed(11)
    n = 50
    base = 2340 + np.cumsum(np.random.randn(n) * 2)
    for i in range(n):
        o = base[i]
        c = base[i] + np.random.randn() * 1.5
        h = max(o, c) + 0.8
        l = min(o, c) - 0.8
        draw_candle(ax1, i, o, c, h, l, w=0.7)

    # Zone S/R cu diferite TF-uri
    zones = [
        (2355, 2358, ['H4', 'D1'],  RED,    'RESIST H4+D1'),
        (2346, 2349, ['M15','H1','H4'], ORANGE, 'RESIST M15+H1+H4 PUTERNIC'),
        (2330, 2333, ['H1', 'H4'],  TEAL,   'SUPORT H1+H4'),
        (2320, 2323, ['M15','H1'],  BLUE,   'SUPORT M15+H1'),
    ]
    for z_lo, z_hi, tfs, col, label in zones:
        ax1.axhspan(z_lo, z_hi, alpha=0.22, color=col)
        ax1.axhline(z_hi, color=col, lw=1.0, linestyle='--', alpha=0.7)
        tf_str = '+'.join(tfs)
        ax1.text(51, (z_lo+z_hi)/2, f'{label}\n({tf_str})', color=col,
                 fontsize=6.5, va='center')

    # Pret actual
    ax1.axhline(2340, color=LIGHT, lw=1.5, linestyle='-', alpha=0.5, label='Pret actual')
    ax1.text(-1.5, 2340, 'Pret\nactual', color=LIGHT, fontsize=7, va='center')

    # Pending order arrow
    ax1.annotate('SELL_LIMIT\nla zona H4+D1',
                 xy=(25, 2356.5), xytext=(30, 2361),
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
                 color=RED, fontsize=7.5, fontweight='bold', ha='center',
                 bbox=dict(fc=BG3, ec=RED, pad=3, boxstyle='round,pad=0.3'))

    ax1.set_xlim(-2, 55)
    ax1.set_ylim(2315, 2367)
    ax1.set_xlabel('Bare', fontsize=7.5, color=GRAY)
    ax1.set_ylabel('XAUUSD', fontsize=7.5, color=GRAY)
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.3)

    # ── Formula confidence ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 2])
    ax2.set_facecolor(BG3)
    ax2.axis('off')
    ax2.text(0.5, 0.98, 'FORMULA CONFIDENCE', color=CYAN, fontsize=8.5,
             fontweight='bold', ha='center', transform=ax2.transAxes)

    lines = [
        ('conf = 25', LIGHT),
        ('  + min(tf_count,4) × 10', TEAL),
        ('  + min(touches,5) × 2', TEAL),
        ('  + fresh_bonus', GREEN),
        ('  − dist_penalty', RED),
        ('', GRAY),
        ('fresh_bonus =', GRAY),
        ('  max(0, 10-age×0.15)', GREEN),
        ('dist_penalty =', GRAY),
        ('  min(20, dist_atr×8)', RED),
        ('', GRAY),
        ('Exemplu zona H4+D1:', YELLOW),
        ('tf=2, touch=3, age=1', GRAY),
        ('dist_atr=0.9 ATR', GRAY),
        ('= 25+20+6+9.85-7.2', GRAY),
        ('= 53.7%', CYAN),
    ]
    y = 0.87
    for txt, col in lines:
        ax2.text(0.06, y, txt, color=col, fontsize=7.5, transform=ax2.transAxes,
                 va='top', family='monospace')
        y -= 0.056

    # ── TF weight panel ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 2])
    ax3.set_facecolor(BG3)
    ax3.axis('off')
    ax3.text(0.5, 0.98, 'IMPORTANTA TF-URILOR', color=CYAN, fontsize=8.5,
             fontweight='bold', ha='center', transform=ax3.transAxes)

    tf_data = [
        ('M15', 1, GRAY),
        ('H1',  2, BLUE),
        ('H4',  3, TEAL),
        ('D1',  4, GREEN),
    ]
    ax3_inner = fig.add_axes([0.73, 0.13, 0.22, 0.22])
    ax3_inner.set_facecolor(BG3)
    tfs_labels = [t[0] for t in tf_data]
    tfs_vals   = [t[1] for t in tf_data]
    tfs_cols   = [t[2] for t in tf_data]
    ax3_inner.barh(tfs_labels, tfs_vals, color=tfs_cols, height=0.5)
    ax3_inner.set_facecolor(BG3)
    ax3_inner.tick_params(labelsize=8, colors=LIGHT)
    ax3_inner.set_xlabel('Greutate relativa', color=GRAY, fontsize=7)
    ax3_inner.spines['top'].set_visible(False)
    ax3_inner.spines['right'].set_visible(False)
    ax3_inner.spines['bottom'].set_color(GRAY2)
    ax3_inner.spines['left'].set_color(GRAY2)
    ax3_inner.text(0.5, 1.08, 'Importanta TF', ha='center',
                   transform=ax3_inner.transAxes, color=GRAY, fontsize=7)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_bollinger_rsi(pdf):
    fig = new_fig()
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.6, wspace=0.45,
                            top=0.95, bottom=0.06, left=0.07, right=0.97)

    ax_hdr = fig.add_subplot(gs[0, :])
    section_title(ax_hdr,
        '5–9. STRATEGII STANDARD',
        'Bollinger, RSI Divergence, London/NY Breakout, Engulfing, Ichimoku',
        color=CYAN)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0:2])
    ax1.set_facecolor(BG2)
    ax1.set_title('Bollinger Bands — Mean Reversion (EURUSD H1)', color=TEAL, fontsize=8, pad=4)

    np.random.seed(5)
    n = 40
    base = 1.0850 + np.cumsum(np.random.randn(n) * 0.0008)
    window = 15
    ma = np.convolve(base, np.ones(window)/window, mode='same')
    std = np.array([np.std(base[max(0,i-window):i+1]) for i in range(n)])
    ub  = ma + 2*std
    lb  = ma - 2*std

    xs = np.arange(n)
    ax1.plot(xs, base, color=LIGHT,  lw=1, alpha=0.9)
    ax1.plot(xs, ma,   color=YELLOW, lw=1.3, linestyle='--', label='Media BB')
    ax1.plot(xs, ub,   color=GRAY,   lw=1.0, label='Banda sus (2σ)')
    ax1.plot(xs, lb,   color=GRAY,   lw=1.0, label='Banda jos (2σ)')
    ax1.fill_between(xs, lb, ub, alpha=0.08, color=GRAY)

    # Touch-uri la banda
    touch_idx = [i for i in range(5, n-2) if base[i] <= lb[i] + 0.0003]
    for ti in touch_idx[:3]:
        ax1.plot(ti, lb[ti], 'o', color=GREEN, ms=8, zorder=5)
        ax1.annotate('BUY\n(RSI<35)', xy=(ti, lb[ti]),
                     xytext=(ti+2, lb[ti]-0.0015),
                     arrowprops=dict(arrowstyle='->', color=GREEN, lw=1),
                     color=GREEN, fontsize=6.5, ha='center')

    ax1.legend(fontsize=6.5, loc='upper left', framealpha=0.5,
               facecolor=BG3, edgecolor=GRAY2)
    ax1.set_xlabel('Bare H1', fontsize=7, color=GRAY)
    ax1.set_ylabel('EURUSD', fontsize=7, color=GRAY)
    ax1.tick_params(labelsize=6.5)
    ax1.grid(True, alpha=0.3)

    # ── RSI Divergence ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2, 0:2])
    ax2.set_facecolor(BG2)
    ax2.set_title('RSI Divergence — Bullish (XAUUSD D1)', color=PINK, fontsize=8, pad=4)

    n2 = 30
    # Pret scade, RSI creste = divergenta bullish
    price2 = np.array([100 - i*0.3 + np.random.randn()*0.4 for i in range(n2)])
    price2[10] = 96.8   # pivot low 1
    price2[22] = 94.5   # pivot low 2 (mai jos)
    rsi2 = np.array([50 - i*0.2 + np.random.randn()*2 for i in range(n2)])
    rsi2[10] = 32   # RSI pivot 1
    rsi2[22] = 38   # RSI pivot 2 (mai sus = divergenta)

    xs2 = np.arange(n2)
    ax2.plot(xs2, price2, color=LIGHT, lw=1.3, label='Pret')

    # Pivot lows
    ax2.plot([10, 22], [price2[10], price2[22]], 'v', color=RED, ms=8, zorder=5)
    ax2.annotate('', xy=(22, price2[22]), xytext=(10, price2[10]),
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
    ax2.text(16, min(price2[10], price2[22])-0.4, 'LL pret',
             color=RED, fontsize=7.5, ha='center', fontweight='bold')

    ax2.set_xlabel('Bare D1', fontsize=7, color=GRAY)
    ax2.set_ylabel('Pret', fontsize=7, color=GRAY)
    ax2.tick_params(labelsize=6.5)
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.set_facecolor(BG2)
    ax2b.plot(xs2, rsi2, color=PINK, lw=1.5, linestyle=':', alpha=0.9, label='RSI')
    ax2b.plot([10, 22], [rsi2[10], rsi2[22]], '^', color=GREEN, ms=8, zorder=5)
    ax2b.annotate('', xy=(22, rsi2[22]), xytext=(10, rsi2[10]),
                  arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))
    ax2b.text(16, max(rsi2[10], rsi2[22])+1.2, 'HL RSI\n→ DIVERGENTA BULL',
              color=GREEN, fontsize=7.5, ha='center', fontweight='bold')
    ax2b.set_ylabel('RSI', fontsize=7, color=PINK)
    ax2b.tick_params(labelsize=6.5, colors=PINK)
    ax2b.axhline(35, color=PINK, lw=0.8, linestyle=':', alpha=0.6)
    ax2b.set_ylim(20, 80)

    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax2b.get_legend_handles_labels()
    ax2.legend(lines2+lines3, labels2+labels3, fontsize=6.5, loc='lower left',
               framealpha=0.5, facecolor=BG3, edgecolor=GRAY2)

    # ── Tabel comparativ strategii ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1:, 2])
    ax3.set_facecolor(BG3)
    ax3.axis('off')
    ax3.text(0.5, 0.99, 'COMPARATIE STRATEGII', color=TEAL, fontsize=8,
             fontweight='bold', ha='center', transform=ax3.transAxes)

    headers = ['Strategie', 'TF', 'Stil', 'ADX']
    data = [
        ('Bollinger',     'H1/H4',  'Range',   '<25'),
        ('RSI Divergence','H4/D1',  'Reversal','Orice'),
        ('London Break.', 'M15/H1', 'Breakout','06-09'),
        ('NY Breakout',   'M15/H1', 'Breakout','13-16'),
        ('China Session', 'M15',    'Range',   '00-06'),
        ('Engulfing',     'H4/D1',  'Reversal','Nivel'),
        ('Ichimoku',      'H4/D1',  'Trend',   'Lent'),
    ]
    colors_tbl = [CYAN, PINK, ORANGE, ORANGE, BLUE, PINK, CYAN]

    y = 0.91
    # Headers
    for j, h in enumerate(headers):
        ax3.text(0.03 + j*0.24, y, h, color=GRAY, fontsize=6.5,
                 transform=ax3.transAxes, fontweight='bold', va='top')
    y -= 0.05
    ax3.plot([0.01, 0.99], [0.86, 0.86], color=GRAY2, lw=0.7, transform=ax3.transAxes, zorder=2)

    for i, (row, col) in enumerate(zip(data, colors_tbl)):
        y = 0.84 - i * 0.11
        # Background alternate
        if i % 2 == 0:
            rect = mpatches.Rectangle((0.01, y-0.06), 0.98, 0.09,
                fc=BG2, ec='none', transform=ax3.transAxes)
            ax3.add_patch(rect)
        for j, val in enumerate(row):
            c = col if j == 0 else LIGHT
            ax3.text(0.03 + j*0.24, y, val, color=c, fontsize=7,
                     transform=ax3.transAxes, va='top',
                     fontweight='bold' if j == 0 else 'normal')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_london_breakout(pdf):
    fig = new_fig()
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.4,
                            top=0.95, bottom=0.06, left=0.07, right=0.97)

    ax_hdr = fig.add_subplot(gs[0, :])
    section_title(ax_hdr,
        '6. LONDON BREAKOUT — Range Asian → Breakout Londra',
        'Range 00-06 UTC se sparge la deschiderea Londrei (06:00 UTC)',
        color=ORANGE)

    # ── Grafic London breakout ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1:, 0])
    ax1.set_facecolor(BG2)
    ax1.set_title('GBPUSD M15 — London Breakout BUY', color=ORANGE, fontsize=9, pad=5)

    # Sesiunea Asiatica: range ingust
    asian = [(1.2625, 1.2630, 1.2638, 1.2618),
             (1.2630, 1.2622, 1.2635, 1.2620),
             (1.2622, 1.2628, 1.2634, 1.2619),
             (1.2628, 1.2624, 1.2632, 1.2621),
             (1.2624, 1.2627, 1.2631, 1.2620),
             (1.2627, 1.2623, 1.2630, 1.2619),
             (1.2623, 1.2629, 1.2633, 1.2620),
             (1.2629, 1.2626, 1.2635, 1.2624)]

    # Breakout Londra
    london = [(1.2626, 1.2648, 1.2655, 1.2624),  # breakout candle
              (1.2648, 1.2662, 1.2665, 1.2645),
              (1.2662, 1.2658, 1.2668, 1.2654),
              (1.2658, 1.2672, 1.2675, 1.2655),
              (1.2672, 1.2681, 1.2685, 1.2669),
              (1.2681, 1.2678, 1.2686, 1.2675),
              (1.2678, 1.2690, 1.2695, 1.2675)]

    all_candles = asian + london
    for i, (o, c, h, l) in enumerate(all_candles):
        draw_candle(ax1, i, o, c, h, l, w=0.55)

    # Asian range box
    asian_high = max(c[2] for c in asian)
    asian_low  = min(c[3] for c in asian)
    ax1.axhspan(asian_low, asian_high, alpha=0.15, color=BLUE, zorder=1)
    ax1.axhline(asian_high, color=BLUE, lw=1.2, linestyle='--', alpha=0.8)
    ax1.axhline(asian_low,  color=BLUE, lw=1.2, linestyle='--', alpha=0.8)
    ax1.text(-1.2, (asian_low+asian_high)/2, 'Range\nAsian', color=BLUE,
             fontsize=7.5, va='center', ha='right')

    # Separator sesiuni
    ax1.axvline(7.5, color=ORANGE, lw=2, linestyle='-', alpha=0.7)
    ax1.text(7.5, asian_low - 0.0008, '06:00\nUTC', color=ORANGE,
             fontsize=7.5, ha='center', va='top', fontweight='bold')
    ax1.text(2, asian_high + 0.0005, 'Sesiunea Asiatica', color=BLUE,
             fontsize=7.5, ha='center', fontweight='bold')
    ax1.text(11, asian_high + 0.0005, 'Sesiunea Londra', color=ORANGE,
             fontsize=7.5, ha='center', fontweight='bold')

    # Breakout candle annotation
    ax1.annotate('BREAKOUT!\ncorp 80% ✓', xy=(8, all_candles[8][1]),
                 xytext=(10.5, 1.2641),
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
                 color=GREEN, fontsize=8, fontweight='bold', ha='center',
                 bbox=dict(fc=BG3, ec=GREEN, pad=3, boxstyle='round,pad=0.3'))

    ax1.set_xlim(-2, 16)
    ax1.set_ylim(1.2610, 1.2705)
    ax1.set_xlabel('Bare M15', fontsize=7.5, color=GRAY)
    ax1.set_ylabel('GBPUSD', fontsize=7.5, color=GRAY)
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.3)

    # ── Filtre si scoring ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor(BG3)
    ax2.axis('off')
    ax2.text(0.5, 0.98, 'FILTRE LONDON BREAKOUT', color=ORANGE, fontsize=8.5,
             fontweight='bold', ha='center', transform=ax2.transAxes)

    filtre = [
        ('session_gate',  'Activ 06:00-09:00 UTC',       True),
        ('asian_range',   'Range 15%-120% din ATR zilnic',True),
        ('breakout',      'Corp lumanare > 60% din range',True),
        ('retest',        'Optional: retest la nivel',     True),
        ('h4_trend',      'EMA20 > EMA50 pe H4 aliniat',  True),
        ('time_window',   'Prime time 06-07:30 +bonus',   True),
        ('day_of_week',   'Mar-Joi +1 / Luni -2 / Vin -1',True),
    ]
    for i, (key, desc, ok) in enumerate(filtre):
        gate_row(ax2, i, f'{key}: {desc}', ok, y_start=0.87, gap=0.11)

    # ── Perechi si sesiuni ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.set_facecolor(BG3)
    ax3.axis('off')
    ax3.text(0.5, 0.98, 'SESIUNI / PERECHI', color=ORANGE, fontsize=8.5,
             fontweight='bold', ha='center', transform=ax3.transAxes)

    sesiuni_info = [
        ('London',    '06:00-09:00 UTC', 'GBPUSD, EURUSD, GBPJPY', ORANGE),
        ('NY',        '13:30-14:30 UTC', 'USDJPY, USDCAD, EURUSD', BLUE),
        ('China',     '00:00-06:00 UTC', 'USDJPY, AUDUSD, NZDUSD', TEAL),
    ]
    y = 0.87
    for name, hours, pairs, col in sesiuni_info:
        ax3.text(0.05, y, f'► {name}', color=col, fontsize=8.5,
                 transform=ax3.transAxes, va='top', fontweight='bold')
        ax3.text(0.05, y-0.09, f'  {hours}', color=GRAY, fontsize=7.5,
                 transform=ax3.transAxes, va='top')
        ax3.text(0.05, y-0.18, f'  {pairs}', color=LIGHT, fontsize=7,
                 transform=ax3.transAxes, va='top')
        y -= 0.32

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_combined_scalp(pdf):
    fig = new_fig()
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.45,
                            top=0.95, bottom=0.06, left=0.05, right=0.97)

    ax_hdr = fig.add_subplot(gs[0, :])
    section_title(ax_hdr,
        '10–11. COMBINED MODE & SCALP BOOST',
        'Combined: vot majoritar >= 60% + spread 21%  |  Scalp: loturi × mult, TP 1:1, max 30 min',
        color=YELLOW)

    # ── Combined Mode diagram ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0:2])
    ax1.set_facecolor(BG2)
    ax1_inner = fig.add_axes([0.07, 0.52, 0.60, 0.30])
    ax1_inner.set_facecolor(BG2)
    ax1_inner.axis('off')
    ax1_inner.set_title('Combined Mode — Vot EURUSD', color=YELLOW, fontsize=9, pad=4)

    # Voturile strategiilor
    strats = ['TrendRider', 'EOB', 'SMC', 'SR_MTF', 'Bollinger']
    votes  = ['BUY', 'BUY', 'HOLD', 'BUY', 'SELL']
    convs  = [7, 9, 0, 6, 3]
    cols_v = [GREEN if v == 'BUY' else (RED if v == 'SELL' else GRAY) for v in votes]

    for i, (s, v, c, col) in enumerate(zip(strats, votes, convs, cols_v)):
        x = i * 0.19 + 0.05
        # Box
        rect = mpatches.FancyBboxPatch((x, 0.5), 0.16, 0.4,
            boxstyle='round,pad=0.02', fc=BG3, ec=col, lw=1.5,
            transform=ax1_inner.transAxes)
        ax1_inner.add_patch(rect)
        ax1_inner.text(x+0.08, 0.78, v, color=col, fontsize=9, ha='center',
                       transform=ax1_inner.transAxes, fontweight='bold')
        ax1_inner.text(x+0.08, 0.62, s, color=GRAY, fontsize=6.5, ha='center',
                       transform=ax1_inner.transAxes)
        if c > 0:
            ax1_inner.text(x+0.08, 0.52, f'conv:{c}', color=YELLOW, fontsize=6,
                           ha='center', transform=ax1_inner.transAxes)

    # Rezultat
    ax1_inner.text(0.5, 0.32, 'BUY 3/5 = 60% ✓   Spread = 40% ✓', color=GREEN,
                   fontsize=9, ha='center', transform=ax1_inner.transAxes,
                   fontweight='bold',
                   bbox=dict(fc=BG3, ec=GREEN, pad=4, boxstyle='round,pad=0.3'))
    ax1_inner.text(0.5, 0.10, '→ SEMNAL COMBINAT: BUY  (SL/TP din EOB conv=9)',
                   color=TEAL, fontsize=8.5, ha='center', transform=ax1_inner.transAxes)

    # ── Scalp Boost diagram ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 2])
    ax2.set_facecolor(BG3)
    ax2.axis('off')
    ax2.text(0.5, 0.98, '⚡ SCALP BOOST MODE', color=ORANGE, fontsize=9,
             fontweight='bold', ha='center', transform=ax2.transAxes)

    scalp_info = [
        ('Loturi ×',    '2.0',    '(risc dublu)', ORANGE),
        ('TP ratio',    '1:1',    '(TP = SL)',     YELLOW),
        ('Max hold',    '30 min', '(exit auto)',   TEAL),
        ('Min. acord',  '3 strat','(din active)',  BLUE),
        ('', '', '', GRAY),
        ('Ex: risc normal = $50', '', '', GRAY),
        ('Scalp: $50 × 2 = $100', '', '', ORANGE),
        ('Win +$100 / Loss -$100', '', '', LIGHT),
        ('Win rate necesar: >50%', '', '', YELLOW),
    ]
    y = 0.88
    for a, b, c, col in scalp_info:
        if not a:
            y -= 0.04
            continue
        ax2.text(0.05, y, a, color=GRAY, fontsize=8, transform=ax2.transAxes, va='top')
        ax2.text(0.55, y, b, color=col, fontsize=8.5, transform=ax2.transAxes,
                 va='top', fontweight='bold')
        ax2.text(0.72, y, c, color=GRAY, fontsize=7.5, transform=ax2.transAxes, va='top')
        y -= 0.095

    # ── Scoring system overview ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_facecolor(BG2)
    ax3.axis('off')
    ax3.text(0.5, 0.97, 'SISTEMUL DE SCORING — Confidence % per Strategie',
             color=TEAL, fontsize=9, fontweight='bold', ha='center',
             transform=ax3.transAxes)

    scoring_data = [
        ('EOB + Unicorn',  85, 20, PURPLE),
        ('SMC v3 STRICT',  91, 12, BLUE),
        ('TrendRider',     78, 10, GREEN),
        ('SR_MTF',         63, 20, CYAN),
        ('Bollinger',      72, 15, CYAN),
        ('London Breakout',82, 12, ORANGE),
    ]
    for j, (name, pct, maxs, col) in enumerate(scoring_data):
        x0 = 0.02 + j * 0.163
        bax = fig.add_axes([x0, 0.06, 0.14, 0.08])
        bax.set_facecolor(BG3)
        bax.set_xlim(0, 100); bax.set_ylim(0, 1)
        bax.axis('off')
        bax.barh(0.35, 100, height=0.35, color=GRAY2)
        bax.barh(0.35, pct, height=0.35, color=col)
        bax.axvline(85, color=YELLOW, lw=1.2, linestyle='--', alpha=0.7)
        bax.text(pct/2, 0.35, f'{pct}%', color=BG, fontsize=8,
                 ha='center', va='center', fontweight='bold')
        bax.text(50, -0.2, name, color=col, fontsize=6.5, ha='center', va='top',
                 fontweight='bold')
        bax.text(86, 0.7, '85%', color=YELLOW, fontsize=5.5)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_ftmo_faq(pdf):
    fig = new_fig()
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.4,
                            top=0.95, bottom=0.05, left=0.05, right=0.97)

    ax_hdr = fig.add_subplot(gs[0, :])
    section_title(ax_hdr,
        '16–17. SETARI FTMO 10K & FAQ',
        'Configuratii sigure pentru challenge + intrebari frecvente',
        color=TEAL)

    # ── FTMO rules visual ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor(BG3)
    ax1.axis('off')
    ax1.text(0.5, 0.98, 'REGULI FTMO 10K', color=TEAL, fontsize=9,
             fontweight='bold', ha='center', transform=ax1.transAxes)

    ftmo_rules = [
        ('Pierdere zilnica max', '$500',  '5% din cont',  RED),
        ('Pierdere totala max',  '$1000', '10% din cont', RED),
        ('Profit target',        '$1000', '10% (faza 1)', GREEN),
        ('Profit target',        '$500',  '5% (faza 2)',  GREEN),
        ('Risc per trade',       '$50',   '0.5% recomandat', ORANGE),
        ('Max pozitii',          '3',     'simultan', TEAL),
        ('Max lot global',       '1.0',   'lot per trade', YELLOW),
    ]
    y = 0.87
    for label, val, sub, col in ftmo_rules:
        ax1.text(0.04, y, label, color=GRAY, fontsize=7.5,
                 transform=ax1.transAxes, va='top')
        ax1.text(0.62, y, val, color=col, fontsize=8.5,
                 transform=ax1.transAxes, va='top', fontweight='bold')
        ax1.text(0.77, y, sub, color=GRAY2, fontsize=6.5,
                 transform=ax1.transAxes, va='top')
        y -= 0.104

    # ── Drawdown chart ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor(BG2)
    ax2.set_title('Expected Value per Trade (Config Sniper)', color=TEAL, fontsize=8, pad=4)

    scenarios = ['Win\n(60% prob)', 'Loss\n(40% prob)', 'Expected\nValue']
    values    = [50*1.5, -50, 50*1.5*0.6 - 50*0.4]  # 45, -50, 25
    bar_cols  = [GREEN, RED, TEAL]
    bars = ax2.bar(scenarios, values, color=bar_cols, width=0.5,
                   edgecolor=GRAY2, linewidth=0.5)
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, v + (1 if v >= 0 else -2.5),
                 f'${v:.0f}', color=LIGHT, ha='center', fontsize=9,
                 fontweight='bold')
    ax2.axhline(0, color=GRAY2, lw=1)
    ax2.set_ylabel('USD', fontsize=8, color=GRAY)
    ax2.tick_params(labelsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0.5, 0.92, 'Risc $50, RR 1:1.5, Win rate 60%',
             color=GRAY, fontsize=7, ha='center', transform=ax2.transAxes)

    # ── FAQ ───────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_facecolor(BG)
    ax3.axis('off')
    ax3.text(0.5, 0.99, 'INTREBARI FRECVENTE (FAQ)', color=TEAL, fontsize=9,
             fontweight='bold', ha='center', transform=ax3.transAxes)

    faqs = [
        ('De ce HOLD cu confidence 59%?',
         'Confidence > 0 = semnal de calitate X dar sub pragul 85%. E comportament CORECT.',
         YELLOW, LIGHT),
        ('Eroare MT5: 10017?',
         'AutoTrading dezactivat in MT5. Click buton verde "AutoTrading" sau Tools→Options→EA→Allow.',
         RED, LIGHT),
        ('TrendRider nu genereaza semnale?',
         'Necesita ADX > 25 + toate 5 conditii aliniate. In range-uri = HOLD mereu. E corect.',
         GREEN, LIGHT),
        ('EOB vs SMC — cand coincid?',
         'Cand ambele spun BUY/SELL pe acelasi simbol = semnal exceptional, intrati cu incredere.',
         BLUE, LIGHT),
        ('Scalp Boost nu executa?',
         'Verificati: Auto Execute ON + MT5 AutoTrading ON + minim 3 strategii premium active.',
         ORANGE, LIGHT),
        ('Signale duplicate in log?',
         'FIX aplicat: dedup de 8 intrari. Acelasi symbol+strategy+signal nu mai apare repetat.',
         TEAL, LIGHT),
    ]

    cols_per_row = 3
    for i, (q, a, qcol, acol) in enumerate(faqs):
        row = i // cols_per_row
        col = i % cols_per_row
        x = col / cols_per_row + 0.01
        y = 0.88 - row * 0.44
        w = 0.31

        # Question box
        rect = mpatches.FancyBboxPatch((x, y-0.35), w, 0.35,
            boxstyle='round,pad=0.01', fc=BG3, ec=qcol, lw=1,
            transform=ax3.transAxes)
        ax3.add_patch(rect)
        ax3.text(x + 0.01, y - 0.03, 'Q: ' + q, color=qcol,
                 fontsize=7, transform=ax3.transAxes, va='top',
                 fontweight='bold', wrap=True)
        ax3.text(x + 0.01, y - 0.14, 'A: ' + a, color=acol,
                 fontsize=6.5, transform=ax3.transAxes, va='top',
                 wrap=True)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_summary(pdf):
    fig = new_fig()
    ax  = fig.add_axes([0, 0, 1, 1])
    ax_off(ax)

    ax.text(0.5, 0.94, 'REZUMAT RAPID — Ce sa activezi in functie de obiectiv',
            color=LIGHT, fontsize=16, fontweight='bold', ha='center',
            transform=ax.transAxes)
    ax.axhline(0.91, xmin=0.05, xmax=0.95, color=TEAL, lw=1.5, alpha=0.7)

    configs = [
        ('FTMO Challenge\n(cat mai sigur)',
         ['EOB + SMC + TrendRider', 'min_confidence: 85%',
          'Risc: $30-50/trade', 'Max 3 pozitii simultane', 'Preset: Sniper'],
         TEAL, 0.07),
        ('Trade-uri Scurte\n(1-2 ore)',
         ['EOB + SMC + TrendRider', 'Scalp Boost: ON',
          'Loturi × 1.5', 'Max hold: 45 min', 'TP ratio: 1:1'],
         ORANGE, 0.32),
        ('Breakout Londra\n(06-09 UTC)',
         ['London Breakout', 'EOB + SR_MTF',
          'Perechi: GBPUSD, EURUSD', 'XAUUSD optional', 'TP ratio: 1:2'],
         ORANGE, 0.57),
        ('Trend Lung\n(H4/D1)',
         ['TrendRider + SMC', 'Ichimoku + RSI Divergence',
          'TF-uri: H4/D1', 'TP ratio: 2.0-2.5', 'Min conf: 80%'],
         GREEN, 0.82),
    ]

    for title, items, col, x in configs:
        bax = fig.add_axes([x, 0.38, 0.22, 0.47])
        bax.set_facecolor(BG2)
        bax.axis('off')

        rect = mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
            boxstyle='round,pad=0.02', fc=BG2, ec=col, lw=2,
            transform=bax.transAxes)
        bax.add_patch(rect)

        bax.text(0.5, 0.94, title, color=col, fontsize=9, ha='center',
                 transform=bax.transAxes, fontweight='bold', va='top')
        bax.plot([0.01, 0.99], [0.84, 0.84], color=col, lw=1, alpha=0.5, transform=bax.transAxes, zorder=2)

        for j, item in enumerate(items):
            bax.text(0.08, 0.78 - j * 0.145, '• ' + item, color=LIGHT,
                     fontsize=8, transform=bax.transAxes, va='top')

    # Bottom metrics
    metrics = [
        ('Expected Value\nper Trade', '+$12-25', GREEN),
        ('Semnale pe zi\n(3 prem. active)', '1-4', TEAL),
        ('Win rate estimat\n(85% conf)', '60-65%', TEAL),
        ('Max drawdown\nrecomandat', '$150/zi', ORANGE),
    ]
    for i, (label, val, col) in enumerate(metrics):
        bax = fig.add_axes([0.07 + i * 0.23, 0.10, 0.19, 0.22])
        bax.set_facecolor(BG3)
        bax.axis('off')
        bax.text(0.5, 0.82, val, color=col, fontsize=18, ha='center',
                 transform=bax.transAxes, fontweight='bold')
        bax.text(0.5, 0.30, label, color=GRAY, fontsize=7.5, ha='center',
                 transform=bax.transAxes, va='top')

    ax.text(0.5, 0.04,
            f'ChartVisualizer AutoTrader v3  |  {datetime.now().strftime("%B %Y")}  '
            f'|  EOB v3 + SMC v3 + TrendRider + SR_MTF + 7 standard',
            color=GRAY2, fontsize=8, ha='center', transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(__file__), 'STRATEGII.pdf')
    print(f'Generez {out} ...')

    with PdfPages(out) as pdf:
        # Metadata
        d = pdf.infodict()
        d['Title']    = 'ChartVisualizer AutoTrader — Ghid Strategii'
        d['Author']   = 'ChartVisualizer v3'
        d['Subject']  = 'Documentatie completa strategii de tranzactionare'
        d['Keywords'] = 'EOB SMC TrendRider SR_MTF trading forex'

        page_cover(pdf)
        page_toc(pdf)
        page_eob(pdf)
        page_smc(pdf)
        page_trend_rider(pdf)
        page_srmtf(pdf)
        page_bollinger_rsi(pdf)
        page_london_breakout(pdf)
        page_combined_scalp(pdf)
        page_ftmo_faq(pdf)
        page_summary(pdf)

    size_mb = os.path.getsize(out) / (1024*1024)
    print(f'Done! {out}  ({size_mb:.1f} MB, 11 pagini)')
