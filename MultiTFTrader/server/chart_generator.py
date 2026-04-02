"""
chart_generator.py
==================
Genereaza grafice multi-timeframe pentru fiecare decizie de tranzactionare.

Graficul contine 4 panouri:
  - H4: context / bias (trend general)
  - D1: tendinta zilnica
  - H1: zona de intrare precisa
  - M5: confirmare pe termen scurt

Pe fiecare panou se afiseaza:
  - Lumanari japoneze (candlesticks)
  - Zone support/rezistenta (shaded bands)
  - Nivele Fibonacci (61.8% OTE, 78.6%, 50%, 38.2%)
  - Order Blocks (dreptunghiuri colorate)
  - Fair Value Gaps (dreptunghiuri transparente)
  - EMA 50 si EMA 200
  - Nivele Entry / SL / TP1 / TP2 (pe H1)
  - Scor confluenta si motive ale deciziei
"""

import matplotlib
matplotlib.use('Agg')  # backend non-interactiv (fara GUI)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os
from datetime import datetime


# ─── Paleta de culori (tema dark) ────────────────────────────────────────────
DARK_BG     = '#0d1117'
PANEL_BG    = '#161b22'
BORDER      = '#30363d'
TEXT_COLOR  = '#e6edf3'
MUTED       = '#8b949e'

BULL_COLOR  = '#26a641'   # verde
BEAR_COLOR  = '#f85149'   # rosu
EMA50_COL   = '#f0b429'   # galben
EMA200_COL  = '#58a6ff'   # albastru

FIB_618_COL = '#f0b429'   # OTE principal
FIB_786_COL = '#ff6b35'   # OTE extins
FIB_500_COL = '#888888'
FIB_382_COL = '#666666'
FIB_236_COL = '#444444'

SUP_COL     = '#26a641'   # verde support
RES_COL     = '#f85149'   # rosu rezistenta
OB_BULL_COL = '#26a641'
OB_BEAR_COL = '#f85149'
FVG_BULL_COL = '#58a6ff'
FVG_BEAR_COL = '#d2a8ff'

ENTRY_COL   = '#ffffff'
SL_COL      = '#f85149'
TP1_COL     = '#26a641'
TP2_COL     = '#00ff88'


# ─── Utilitare ────────────────────────────────────────────────────────────────

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def draw_candles(ax, df):
    """Deseneaza candlestick-uri pe axele date."""
    for i, (_, row) in enumerate(df.iterrows()):
        color = BULL_COLOR if row['close'] >= row['open'] else BEAR_COLOR

        # Wick (fitil)
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=0.7, alpha=0.85)

        # Body (corp)
        bot = min(row['open'], row['close'])
        top = max(row['open'], row['close'])
        h   = max(top - bot, 1e-8)

        rect = plt.Rectangle((i - 0.38, bot), 0.76, h, color=color, alpha=0.9, zorder=2)
        ax.add_patch(rect)


def draw_ema(ax, df, period, color, label):
    """Deseneaza EMA pe axele date."""
    if len(df) < period:
        return
    values = ema(df['close'], period)
    ax.plot(range(len(df)), values, color=color, linewidth=0.9,
            linestyle='--', alpha=0.75, label=label, zorder=3)


def draw_fibonacci(ax, fib, n_bars, price_range):
    """Deseneaza nivelele Fibonacci pe axele date."""
    if not fib:
        return

    pmin, pmax = price_range
    levels = [
        ('level_236', FIB_236_COL, '23.6%',  '--', 0.5),
        ('level_382', FIB_382_COL, '38.2%',  '--', 0.6),
        ('level_500', FIB_500_COL, '50.0%',  '-',  0.65),
        ('level_618', FIB_618_COL, '61.8% OTE', '-', 0.85),
        ('level_786', FIB_786_COL, '78.6% OTE', '-', 0.85),
    ]

    for key, color, label, ls, alpha in levels:
        val = fib.get(key)
        if val is None:
            continue
        if not (pmin * 0.998 <= val <= pmax * 1.002):
            continue
        ax.axhline(y=val, color=color, linewidth=0.9, linestyle=ls, alpha=alpha, zorder=4)
        ax.text(n_bars - 0.5, val, f' {label}', color=color, fontsize=6.5,
                va='center', ha='left', zorder=5)

    # Zona OTE hasurata
    l618 = fib.get('level_618')
    l786 = fib.get('level_786')
    if l618 and l786 and pmin <= min(l618, l786) <= pmax:
        ax.axhspan(min(l618, l786), max(l618, l786),
                   alpha=0.08, color=FIB_618_COL, zorder=1)


def draw_sr_zones(ax, zones, n_bars, price_range):
    """Deseneaza zone support/rezistenta."""
    if not zones:
        return

    pmin, pmax = price_range

    for zone in zones.get('support', []):
        lvl = zone.get('level', 0)
        str_ = zone.get('strength', 1)
        if not (pmin * 0.997 <= lvl <= pmax * 1.003):
            continue
        w = lvl * 0.0003 * (1 + str_ * 0.1)
        ax.axhspan(lvl - w, lvl + w, alpha=min(0.12 + str_ * 0.04, 0.3),
                   color=SUP_COL, zorder=1)
        ax.axhline(y=lvl, color=SUP_COL, linewidth=0.7, alpha=0.55, linestyle='--', zorder=3)

    for zone in zones.get('resistance', []):
        lvl = zone.get('level', 0)
        str_ = zone.get('strength', 1)
        if not (pmin * 0.997 <= lvl <= pmax * 1.003):
            continue
        w = lvl * 0.0003 * (1 + str_ * 0.1)
        ax.axhspan(lvl - w, lvl + w, alpha=min(0.12 + str_ * 0.04, 0.3),
                   color=RES_COL, zorder=1)
        ax.axhline(y=lvl, color=RES_COL, linewidth=0.7, alpha=0.55, linestyle='--', zorder=3)


def draw_order_blocks(ax, obs, n_bars, price_range):
    """Deseneaza Order Blocks ca dreptunghiuri colorate."""
    if not obs:
        return

    pmin, pmax = price_range

    for ob in obs[-4:]:
        top    = ob.get('top', 0)
        bottom = ob.get('bottom', 0)
        idx    = ob.get('index', 0)
        typ    = ob.get('type', 'bullish')

        if top <= 0 or bottom <= 0 or top <= bottom:
            continue
        if not (pmin * 0.997 <= bottom <= pmax * 1.003 or
                pmin * 0.997 <= top    <= pmax * 1.003):
            continue

        color = OB_BULL_COL if typ == 'bullish' else OB_BEAR_COL
        width = max(n_bars - idx, 2)

        rect = plt.Rectangle((idx, bottom), width, top - bottom,
                              color=color, alpha=0.13, linewidth=0.8,
                              edgecolor=color, zorder=1)
        ax.add_patch(rect)
        ax.text(idx + 0.5, (top + bottom) / 2, 'OB',
                color=color, fontsize=6, va='center', ha='left', alpha=0.85, zorder=5)


def draw_fvg(ax, fvgs, n_bars, price_range):
    """Deseneaza Fair Value Gaps."""
    if not fvgs:
        return

    pmin, pmax = price_range

    for fvg in fvgs[-4:]:
        top    = fvg.get('top', 0)
        bottom = fvg.get('bottom', 0)
        idx    = fvg.get('index', 0)
        typ    = fvg.get('type', 'bullish')

        if top <= 0 or bottom <= 0 or top <= bottom:
            continue
        if not (pmin * 0.997 <= bottom <= pmax * 1.003 or
                pmin * 0.997 <= top    <= pmax * 1.003):
            continue

        color = FVG_BULL_COL if typ == 'bullish' else FVG_BEAR_COL
        width = max(n_bars - idx, 2)

        rect = plt.Rectangle((idx, bottom), width, top - bottom,
                              color=color, alpha=0.10, linewidth=0.7,
                              edgecolor=color, linestyle='dotted', zorder=1)
        ax.add_patch(rect)
        ax.text(idx + 0.5, (top + bottom) / 2, 'FVG',
                color=color, fontsize=5.5, va='center', ha='left', alpha=0.8, zorder=5)


def draw_entry_levels(ax, decision, n_bars, price_range):
    """Deseneaza Entry, SL, TP1, TP2 (doar pe panoul H1)."""
    pmin, pmax = price_range

    def hline(level, color, label):
        if not level or not (pmin * 0.995 <= level <= pmax * 1.005):
            return
        ax.axhline(y=level, color=color, linewidth=1.6, alpha=0.9, linestyle='-', zorder=6)
        ax.text(n_bars - 0.3, level, f'  {label}: {level:.5f}',
                color=color, fontsize=7.5, va='center', fontweight='bold', zorder=7)

    entry = decision.get('entry_price', 0)
    sl    = decision.get('stop_loss', 0)
    tp1   = decision.get('take_profit_1', 0)
    tp2   = decision.get('take_profit_2', 0)

    hline(entry, ENTRY_COL, 'ENTRY')
    hline(sl,    SL_COL,    'SL')
    hline(tp1,   TP1_COL,   'TP1')
    hline(tp2,   TP2_COL,   'TP2')

    # Zone risc / profit
    if entry and sl:
        ax.axhspan(min(entry, sl), max(entry, sl), alpha=0.06, color=SL_COL, zorder=0)
    if entry and tp1:
        ax.axhspan(min(entry, tp1), max(entry, tp1), alpha=0.06, color=TP1_COL, zorder=0)


def style_axis(ax, title):
    """Aplica tema dark pe o axe."""
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_title(title, color=TEXT_COLOR, fontsize=9, pad=4, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(axis='x', labelbottom=False)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
    ax.yaxis.tick_right()


def plot_panel(ax, df, tf_name, fib, sr_zones, obs, fvgs, decision=None):
    """Deseneaza un singur panou (timeframe)."""
    if df is None or len(df) < 5:
        ax.text(0.5, 0.5, f'Date insuficiente {tf_name}',
                transform=ax.transAxes, ha='center', va='center', color=MUTED)
        style_axis(ax, tf_name)
        return

    df = df.tail(80).reset_index(drop=True)
    n  = len(df)

    pmin = df['low'].min()
    pmax = df['high'].max()
    prng = pmax - pmin
    pad  = prng * 0.06

    draw_candles(ax, df)
    draw_sr_zones(ax, sr_zones, n, (pmin, pmax))
    draw_fibonacci(ax, fib, n, (pmin, pmax))
    draw_order_blocks(ax, obs, n, (pmin, pmax))
    draw_fvg(ax, fvgs, n, (pmin, pmax))
    draw_ema(ax, df, 50,  EMA50_COL,  'EMA50')
    draw_ema(ax, df, 200, EMA200_COL, 'EMA200')

    # Linie pret curent
    ax.axhline(y=df['close'].iloc[-1], color='#ffffff',
               linewidth=0.4, alpha=0.4, linestyle=':', zorder=3)

    # Nivele de tranzactionare (doar pe H1)
    if decision:
        draw_entry_levels(ax, decision, n, (pmin - pad, pmax + pad))

    ax.set_xlim(-1, n + 2)
    ax.set_ylim(pmin - pad, pmax + pad)
    style_axis(ax, tf_name)

    # Mini legenda EMA
    if n >= 50:
        ax.legend(fontsize=6, loc='upper left', facecolor=PANEL_BG,
                  edgecolor=BORDER, labelcolor=MUTED, framealpha=0.7)


# ─── Functia principala ───────────────────────────────────────────────────────

def create_trade_chart(symbol, timeframe_data, analysis, decision, chart_dir="charts"):
    """
    Creeaza graficul multi-timeframe si il salveaza ca PNG.

    Parametri:
        symbol         : e.g. "EURUSD"
        timeframe_data : dict { "H4": pd.DataFrame, "H1": pd.DataFrame, ... }
        analysis       : dict cu cheile:
                            'fibonacci'    -> { tf: fib_dict }
                            'sr_zones'     -> { tf: zones_dict }
                            'order_blocks' -> { tf: [ob, ...] }
                            'fvg'          -> { tf: [fvg, ...] }
        decision       : dict cu: decision, entry_price, stop_loss,
                                  take_profit_1, take_profit_2,
                                  confidence_score, confluence_score,
                                  rr_ratio, reasons
        chart_dir      : directorul unde se salveaza PNG-ul

    Returneaza:
        filename (str) - numele fisierului PNG salvat
    """
    os.makedirs(chart_dir, exist_ok=True)

    fig = plt.figure(figsize=(24, 18))
    fig.patch.set_facecolor(DARK_BG)

    # Layout: 3 randuri x 2 coloane
    # Rand 1: H4 (wide) | H1 (wide)   ← context + intrare
    # Rand 2: M15 (wide) | M5 (wide)  ← confirmare
    # Rand 3: M1 (full width)          ← precizie
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.35,
        wspace=0.06,
        width_ratios=[1, 1],
        height_ratios=[1.2, 1, 0.8],
        left=0.03, right=0.93,
        top=0.90, bottom=0.06
    )

    ax_h4  = fig.add_subplot(gs[0, 0])
    ax_h1  = fig.add_subplot(gs[0, 1])
    ax_m15 = fig.add_subplot(gs[1, 0])
    ax_m5  = fig.add_subplot(gs[1, 1])
    ax_m1  = fig.add_subplot(gs[2, :])   # M1 ocupa toata latimea

    # ── Deseneaza panourile ──
    def get(key, tf):
        return analysis.get(key, {}).get(tf)

    plot_panel(ax_h4, timeframe_data.get('H4'), 'H4 — Bias / Trend',
               get('fibonacci','H4'), get('sr_zones','H4'),
               get('order_blocks','H4'), get('fvg','H4'))

    plot_panel(ax_h1, timeframe_data.get('H1'), 'H1 — Zona Intrare + Entry/SL/TP',
               get('fibonacci','H1'), get('sr_zones','H1'),
               get('order_blocks','H1'), get('fvg','H1'),
               decision=decision)

    plot_panel(ax_m15, timeframe_data.get('M15'), 'M15 — Confirmare',
               get('fibonacci','M15'), get('sr_zones','M15'),
               get('order_blocks','M15'), get('fvg','M15'))

    plot_panel(ax_m5, timeframe_data.get('M5'), 'M5 — Confirmare Fina',
               get('fibonacci','M5'), get('sr_zones','M5'),
               get('order_blocks','M5'), get('fvg','M5'))

    plot_panel(ax_m1, timeframe_data.get('M1'), 'M1 — Precizie Intrare',
               get('fibonacci','M1'), get('sr_zones','M1'),
               get('order_blocks','M1'), get('fvg','M1'))

    # ── Titlu principal ──
    action = decision.get('decision', 'HOLD')
    conf   = decision.get('confidence_score', 0)
    rr     = decision.get('rr_ratio', 0)
    confl  = decision.get('confluence_score', 0)

    title_color = BULL_COLOR if action == 'BUY' else \
                  BEAR_COLOR if action == 'SELL' else MUTED

    fig.suptitle(
        f"{symbol}  ▸  {action}  |  Confidence: {conf}%  |  R:R 1:{rr:.1f}  |  Confluenta: {confl}",
        fontsize=15, fontweight='bold', color=title_color, y=0.96
    )

    # ── Timestamp ──
    ts = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    fig.text(0.92, 0.96, ts, color=MUTED, fontsize=8, ha='right', va='top')

    # ── Legenda simboluri ──
    legend_items = [
        mpatches.Patch(color=BULL_COLOR,    label='Bullish / Support'),
        mpatches.Patch(color=BEAR_COLOR,    label='Bearish / Resistance'),
        mpatches.Patch(color=FIB_618_COL,   label='Fibonacci OTE (61.8%)'),
        mpatches.Patch(color=FVG_BULL_COL,  label='Fair Value Gap'),
        mpatches.Patch(color=ENTRY_COL,     label='Entry'),
        mpatches.Patch(color=SL_COL,        label='Stop Loss'),
        mpatches.Patch(color=TP1_COL,       label='Take Profit'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=7,
               facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=MUTED,
               fontsize=7.5, framealpha=0.85, bbox_to_anchor=(0.5, 0.005))

    # ── Motivele deciziei (partea stanga jos) ──
    reasons = decision.get('reasons', [])
    if reasons:
        reasons_text = '\n'.join([f'• {r}' for r in reasons[:7]])
        fig.text(
            0.04, 0.04, reasons_text,
            fontsize=7.5, color=MUTED, va='bottom', ha='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=PANEL_BG,
                      edgecolor=BORDER, alpha=0.85)
        )

    # ── R:R info (dreapta jos) ──
    entry = decision.get('entry_price', 0)
    sl    = decision.get('stop_loss', 0)
    tp1   = decision.get('take_profit_1', 0)
    risk  = abs(entry - sl)
    rew   = abs(tp1 - entry)

    info_lines = [
        f"Entry  : {entry:.5f}",
        f"SL     : {sl:.5f}  (risc: {risk:.5f})",
        f"TP1    : {tp1:.5f}  (reward: {rew:.5f})",
        f"R:R    : 1:{rr:.2f}",
        f"Risk % : {decision.get('risk_percent', 0.75):.2f}%",
    ]
    fig.text(
        0.93, 0.12, '\n'.join(info_lines),
        fontsize=7.5, color=TEXT_COLOR, va='bottom', ha='left',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=PANEL_BG,
                  edgecolor=title_color, alpha=0.9)
    )

    # ── Salveaza ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = f"{symbol}_{action}_{timestamp}.png"
    filepath  = os.path.join(chart_dir, filename)

    plt.savefig(filepath, dpi=130, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)

    return filename
