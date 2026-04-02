"""
server.py - MultiTFTrader Backend
===================================
Primeste date OHLCV de pe 5 timeframe-uri (M1, M5, H1, H4, D1),
calculeaza:
  - Zone support/rezistenta (clustering de swing highs/lows)
  - Nivele Fibonacci (OTE 61.8% - 78.6%)
  - Order Blocks (ultima lumanare opusa inainte de miscare impulsiva)
  - Fair Value Gaps (goluri de pret intre lumanari non-adiacente)
  - Confluenta multi-timeframe (scor ponderat)

Genereaza un grafic PNG cu toata analiza si returneaza decizia.

Endpoints:
  POST /analyze_mtf  -> analiza principala + generare grafic
  GET  /charts/<file> -> servire grafice PNG
  GET  /charts        -> dashboard HTML cu graficele recente
  GET  /health        -> status server
"""

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import json
import os
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

# Import generatorul de grafice local
from chart_generator import create_trade_chart

# ─── Configurare ─────────────────────────────────────────────────────────────
app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

CHARTS_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# Incarca config
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

SERVER_HOST         = config.get('host', 'localhost')
SERVER_PORT         = config.get('port', 5001)
MIN_CONFLUENCE      = config.get('min_confluence_score', 5)
MIN_RR              = config.get('min_rr', 2.0)
MIN_CONFIDENCE      = config.get('min_confidence', 75)   # 75% precizie minima
NEWS_BLACKOUT_PRE   = config.get('news_blackout_before_min', 30)
NEWS_BLACKOUT_POST  = config.get('news_blackout_after_min', 15)

# Ponderi timeframe — M15 adaugat
TF_WEIGHTS = {'M1': 1, 'M5': 2, 'M15': 3, 'H1': 4, 'H4': 5}

# Cache stiri pentru a nu face prea multe request-uri
_news_cache  = {'data': [], 'fetched_at': None}
_signal_log  = []   # ultimele 200 semnale pentru dashboard /signals
MAX_LOG_SIZE = 200


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESARE DATE
# ═══════════════════════════════════════════════════════════════════════════════

def bars_to_df(bars_list):
    """Converteste lista de bare [time, open, high, low, close, vol] in DataFrame."""
    if not bars_list or len(bars_list) < 5:
        return pd.DataFrame()
    df = pd.DataFrame(bars_list, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('time').sort_index()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series, period=14):
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def calc_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTARE SWING HIGH / LOW
# ═══════════════════════════════════════════════════════════════════════════════

def find_swings(df, lookback=5):
    """
    Gaseste swing highs si swing lows (puncte pivot).
    Un swing high = barul cu highest high din fereastra [i-lookback, i+lookback].
    """
    highs, lows = [], []
    n = len(df)

    for i in range(lookback, n - lookback):
        win_h = df['high'].iloc[i - lookback: i + lookback + 1]
        win_l = df['low'].iloc[i  - lookback: i + lookback + 1]

        if df['high'].iloc[i] == win_h.max():
            highs.append({'price': float(df['high'].iloc[i]), 'index': i})

        if df['low'].iloc[i] == win_l.min():
            lows.append({'price': float(df['low'].iloc[i]), 'index': i})

    return highs, lows


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE SUPPORT / REZISTENTA
# ═══════════════════════════════════════════════════════════════════════════════

def find_sr_zones(df, tolerance_pct=0.0015, lookback=5):
    """
    Grupeaza swing highs/lows in zone S/R.
    Zonele cu mai multe teste sunt mai puternice.
    """
    highs, lows = find_swings(df, lookback)

    def cluster(levels):
        if not levels:
            return []
        prices = [l['price'] for l in levels]
        idxs   = [l['index'] for l in levels]
        used   = [False] * len(prices)
        zones  = []

        for i in range(len(prices)):
            if used[i]:
                continue
            grp_p = [prices[i]]
            grp_i = [idxs[i]]
            used[i] = True

            for j in range(i + 1, len(prices)):
                if not used[j] and abs(prices[j] - prices[i]) / prices[i] < tolerance_pct:
                    grp_p.append(prices[j])
                    grp_i.append(idxs[j])
                    used[j] = True

            zones.append({
                'level':            round(float(np.mean(grp_p)), 8),
                'strength':         len(grp_p),
                'last_touch_index': max(grp_i)
            })

        zones.sort(key=lambda x: x['strength'], reverse=True)
        return zones[:6]

    return {
        'support':    cluster(lows),
        'resistance': cluster(highs)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIBONACCI
# ═══════════════════════════════════════════════════════════════════════════════

def find_fibonacci(df, lookback=5, fib_swing=None):
    """
    Traseaza Fibonacci din ULTIMUL SWING IMPULSIV semnificativ.

    Metoda ICT corecta:
      - Gaseste ultimele doua pivot-uri semnificative (un High si un Low)
      - Le ordoneaza cronologic: care a venit primul si care al doilea
      - Fibonacci se traseaza intre ele (impulsul, nu extremul absolut)
      - OTE zona = 61.8% - 78.6% din acel impuls

    Exemplu:
      Daca ultimul swing low e la index 180 si ultimul swing high la index 200:
      → impuls UP: low→high
      → retracement DOWN → cautam BUY la 61.8%-78.6% din impuls
    """
    highs, lows = find_swings(df, lookback)
    if not highs or not lows:
        return None

    # Ultimul swing high si low semnificativ (cel mai recent)
    last_h = max(highs, key=lambda x: x['index'])
    last_l = max(lows,  key=lambda x: x['index'])

    sh     = last_h['price']
    sl     = last_l['price']
    hi_idx = last_h['index']
    lo_idx = last_l['index']

    rng = sh - sl
    if rng < 1e-8:
        return None

    # Determina directia impulsului cronologic
    if hi_idx > lo_idx:
        # Impuls UP: low a venit inainte, high dupa → retracement bearish (cautam SELL)
        # Fibonacci de la low (0%) la high (100%), OTE la 61.8-78.6% = zona SELL
        direction = 'bullish_impulse'
        fib = {
            'high': sh, 'low': sl,
            'impulse_from': sl, 'impulse_to': sh,
            'direction': direction,
            'level_236': round(sh - 0.236 * rng, 8),
            'level_382': round(sh - 0.382 * rng, 8),
            'level_500': round(sh - 0.500 * rng, 8),
            'level_618': round(sh - 0.618 * rng, 8),  # OTE SELL
            'level_786': round(sh - 0.786 * rng, 8),  # OTE SELL extins
        }
    else:
        # Impuls DOWN: high a venit inainte, low dupa → retracement bullish (cautam BUY)
        # Fibonacci de la high (0%) la low (100%), OTE la 61.8-78.6% = zona BUY
        direction = 'bearish_impulse'
        fib = {
            'high': sh, 'low': sl,
            'impulse_from': sh, 'impulse_to': sl,
            'direction': direction,
            'level_236': round(sl + 0.236 * rng, 8),
            'level_382': round(sl + 0.382 * rng, 8),
            'level_500': round(sl + 0.500 * rng, 8),
            'level_618': round(sl + 0.618 * rng, 8),  # OTE BUY
            'level_786': round(sl + 0.786 * rng, 8),  # OTE BUY extins
        }

    return fib


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════

def find_order_blocks(df):
    """
    Detecteaza Order Blocks:
    - Bullish OB = ultima lumanare bearish inaintea unei miscari bullish impulsive
    - Bearish OB = ultima lumanare bullish inaintea unei miscari bearish impulsive
    """
    if len(df) < 4:
        return []

    obs  = []
    atr  = calc_atr(df)
    body = (df['close'] - df['open']).abs().mean()

    for i in range(1, len(df) - 1):
        c   = df.iloc[i]
        nxt = df.iloc[i + 1]

        # Bullish OB
        if (c['close'] < c['open'] and                         # lumanare bearish
                nxt['close'] > c['high'] and                   # breakout bullish
                (nxt['close'] - nxt['open']) > body * 0.7):    # miscare impulsiva

            obs.append({
                'type':   'bullish',
                'top':    float(c['open']),
                'bottom': float(c['close']),
                'index':  i,
                'time':   str(df.index[i])
            })

        # Bearish OB
        elif (c['close'] > c['open'] and
                  nxt['close'] < c['low'] and
                  (nxt['open'] - nxt['close']) > body * 0.7):

            obs.append({
                'type':   'bearish',
                'top':    float(c['close']),
                'bottom': float(c['open']),
                'index':  i,
                'time':   str(df.index[i])
            })

    return obs[-5:]


# ═══════════════════════════════════════════════════════════════════════════════
# FAIR VALUE GAPS
# ═══════════════════════════════════════════════════════════════════════════════

def find_fvg(df):
    """
    Detecteaza Fair Value Gaps (imbalante de pret):
    - Bullish FVG: low[i+1] > high[i-1]  (gap in sus)
    - Bearish FVG: high[i+1] < low[i-1]  (gap in jos)
    """
    if len(df) < 3:
        return []

    fvgs = []
    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        nxt  = df.iloc[i + 1]

        # Bullish FVG
        if nxt['low'] > prev['high']:
            fvgs.append({
                'type':   'bullish',
                'bottom': float(prev['high']),
                'top':    float(nxt['low']),
                'index':  i,
                'size':   float(nxt['low'] - prev['high']),
                'time':   str(df.index[i])
            })

        # Bearish FVG
        elif nxt['high'] < prev['low']:
            fvgs.append({
                'type':   'bearish',
                'bottom': float(nxt['high']),
                'top':    float(prev['low']),
                'index':  i,
                'size':   float(prev['low'] - nxt['high']),
                'time':   str(df.index[i])
            })

    return fvgs[-5:]


def find_mss(df, lookback=15):
    """
    Market Structure Shift: ultimul close sparge cel mai recent swing high/low.
    Bullish MSS = close > swing high → structura s-a inversat sus (cautam BUY)
    Bearish MSS = close < swing low  → structura s-a inversat jos (cautam SELL)
    """
    if len(df) < lookback + 2:
        return None

    recent   = df.iloc[-(lookback + 1):-1]
    last_bar = df.iloc[-1]

    swing_high = float(recent['high'].max())
    swing_low  = float(recent['low'].min())
    last_close = float(last_bar['close'])

    if last_close > swing_high:
        return 'bullish'
    elif last_close < swing_low:
        return 'bearish'
    return None


def find_displacement(df, lookback=3):
    """
    Displacement: lumanare puternica (corp >= 1.5x medie) care arata momentum institutional.
    Returneaza 'bullish', 'bearish' sau None.
    """
    if len(df) < 20:
        return None

    bodies   = (df['close'] - df['open']).abs()
    avg_body = float(bodies.iloc[-20:-lookback].mean())
    if avg_body < 1e-10:
        return None

    for i in range(-lookback, 0):
        bar  = df.iloc[i]
        body = abs(float(bar['close']) - float(bar['open']))
        rng  = float(bar['high']) - float(bar['low'])

        if body < avg_body * 1.5:
            continue
        if rng > 0 and body / rng < 0.6:
            continue

        return 'bullish' if bar['close'] > bar['open'] else 'bearish'

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# ANALIZA COMPLETA PE UN TIMEFRAME
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_tf(df, fib_swing=None):
    """Ruleaza toata analiza pe un singur DataFrame de timeframe."""
    if df is None or len(df) < 20:
        return {}

    current = float(df['close'].iloc[-1])
    ema50   = float(calc_ema(df['close'], 50).iloc[-1])
    ema200  = float(calc_ema(df['close'], 200).iloc[-1]) if len(df) >= 200 else current
    rsi     = calc_rsi(df['close'])
    atr     = calc_atr(df)

    return {
        'sr_zones':     find_sr_zones(df),
        'fibonacci':    find_fibonacci(df, fib_swing=fib_swing),  # foloseste 1000-bar swing
        'order_blocks': find_order_blocks(df),
        'fvg':          find_fvg(df),
        'mss':          find_mss(df),
        'displacement': find_displacement(df),
        'rsi':          rsi,
        'ema50':        ema50,
        'ema200':       ema200,
        'atr':          atr,
        'current':      current,
        'trend':        'bullish' if current > ema50 else 'bearish',
        'bias':         'bullish' if current > ema200 else 'bearish',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CONFLUENTA MULTI-TIMEFRAME
# ═══════════════════════════════════════════════════════════════════════════════

def check_confluence(price, tf_analysis, digits=5):
    """
    Verifica confluenta zonelor pe toate timeframe-urile.

    Scor:
      - Zona S/R confirmata pe TF: +weight
      - Fibonacci OTE (61.8-78.6%): +weight
      - Order Block activ:          +weight

    Returneaza directia (BUY/SELL/HOLD) si scorul.
    """
    tol = price * 0.0018   # toleranta ±0.18%

    bull_score = 0
    bear_score = 0
    total_score = 0
    max_score = 0
    details = {}
    confluence_zones = []

    for tf, ana in tf_analysis.items():
        if not ana:
            continue
        w = TF_WEIGHTS.get(tf, 1)
        max_score += w * 3
        tf_score = 0
        det = {'zone': False, 'fib': False, 'ob': False, 'fvg': False,
               'trend': ana.get('trend', ''), 'bias': ana.get('bias', '')}

        # ── Zone S/R (max +w o singura data per TF) ──
        zone_scored = False
        for zone in ana.get('sr_zones', {}).get('support', []):
            if abs(zone['level'] - price) <= tol:
                tf_score  += w
                bull_score += w
                det['zone'] = True
                zone_scored = True
                confluence_zones.append({
                    'tf': tf, 'type': 'support',
                    'level': zone['level'], 'strength': zone['strength']
                })
                break

        if not zone_scored:
            for zone in ana.get('sr_zones', {}).get('resistance', []):
                if abs(zone['level'] - price) <= tol:
                    tf_score  += w
                    bear_score += w
                    det['zone'] = True
                    confluence_zones.append({
                        'tf': tf, 'type': 'resistance',
                        'level': zone['level'], 'strength': zone['strength']
                    })
                    break

        # ── Fibonacci OTE ──
        fib = ana.get('fibonacci')
        if fib:
            l618 = fib.get('level_618', 0)
            l786 = fib.get('level_786', 0)
            if l618 and l786:
                f_lo = min(l618, l786) - tol
                f_hi = max(l618, l786) + tol
                if f_lo <= price <= f_hi:
                    tf_score += w
                    det['fib'] = True
                    direction = fib.get('direction', '')
                    if 'bullish' in direction:
                        bull_score += w
                    else:
                        bear_score += w

        # ── Order Blocks ──
        for ob in ana.get('order_blocks', [])[-3:]:
            ob_lo = ob.get('bottom', 0)
            ob_hi = ob.get('top', 0)
            if ob_lo <= price <= ob_hi or abs(price - ob_lo) <= tol or abs(price - ob_hi) <= tol:
                tf_score += w
                det['ob'] = True
                if ob.get('type') == 'bullish':
                    bull_score += w
                else:
                    bear_score += w
                break

        # ── FVG ──
        for fvg in ana.get('fvg', [])[-3:]:
            fvg_lo = fvg.get('bottom', 0)
            fvg_hi = fvg.get('top', 0)
            if fvg_lo <= price <= fvg_hi:
                det['fvg'] = True
                break

        total_score += tf_score
        details[tf] = det

    # ── Bonus trend H4 si H1 (tranzactionam cu trendul, nu contra lui) ──
    for tf, w_bonus in [('H4', 5), ('H1', 4)]:
        ana = tf_analysis.get(tf, {})
        if not ana:
            continue
        if ana.get('bias') == 'bullish':   # pret > EMA200
            bull_score += w_bonus
            total_score += w_bonus
        elif ana.get('bias') == 'bearish':
            bear_score += w_bonus
            total_score += w_bonus
        max_score += w_bonus

    # ── Bonus displacement + MSS (confirmare momentum/structura pe entry TF) ──
    for tf, w_disp in [('M5', 3), ('M15', 4)]:
        disp = tf_analysis.get(tf, {}).get('displacement')
        if disp == 'bullish':
            bull_score += w_disp
            total_score += w_disp
            max_score += w_disp
        elif disp == 'bearish':
            bear_score += w_disp
            total_score += w_disp
            max_score += w_disp

        mss = tf_analysis.get(tf, {}).get('mss')
        if mss == 'bullish':
            bull_score += w_disp
            total_score += w_disp
            max_score += w_disp
        elif mss == 'bearish':
            bear_score += w_disp
            total_score += w_disp
            max_score += w_disp

    # Determina directia
    if bull_score > bear_score * 1.3:
        direction = 'BUY'
    elif bear_score > bull_score * 1.3:
        direction = 'SELL'
    else:
        direction = 'HOLD'

    # Scorul directional (nu total) pentru confidence corect
    direction_score = bull_score if direction == 'BUY' else bear_score if direction == 'SELL' else 0

    return {
        'direction':        direction,
        'score':            total_score,
        'direction_score':  direction_score,
        'max_score':        max_score,
        'bull_score':       bull_score,
        'bear_score':       bear_score,
        'details':          details,
        'confluence_zones': confluence_zones
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CALCULUL NIVELELOR DE TRANZACTIONARE
# ═══════════════════════════════════════════════════════════════════════════════

def calc_trade_levels(price, direction, tf_analysis, digits=5):
    """
    Calculeaza Entry, SL, TP1, TP2 bazat pe ATR si zone S/R.
    """
    h1  = tf_analysis.get('H1', {})
    h4  = tf_analysis.get('H4', {})
    atr = h1.get('atr', price * 0.001) or price * 0.001

    def nearest_zone(zones_list, above=True):
        candidates = [z['level'] for z in zones_list
                      if (z['level'] > price if above else z['level'] < price)]
        if not candidates:
            return None
        return min(candidates) if above else max(candidates)

    h4_sr = h4.get('sr_zones', {})

    def second_nearest_zone(zones_list, above=True, first_level=None):
        """A doua zona dupa prima — pentru TP2 diferit de TP1."""
        candidates = sorted(
            [z['level'] for z in zones_list if (z['level'] > price if above else z['level'] < price)],
            reverse=not above
        )
        if not candidates:
            return None
        if first_level and len(candidates) > 1:
            for lvl in candidates:
                if above and lvl > first_level + price * 0.0005:
                    return lvl
                if not above and lvl < first_level - price * 0.0005:
                    return lvl
        return None

    if direction == 'BUY':
        entry  = price
        sl     = round(entry - atr * 2.0, digits)   # 2.0 ATR — mai mult spatiu
        tp1_sr = nearest_zone(h4_sr.get('resistance', []), above=True)
        tp1    = tp1_sr if tp1_sr else round(entry + atr * 3.0, digits)
        tp2_sr = second_nearest_zone(h4_sr.get('resistance', []), above=True, first_level=tp1)
        tp2    = tp2_sr if tp2_sr else round(entry + atr * 5.0, digits)

    else:  # SELL
        entry  = price
        sl     = round(entry + atr * 2.0, digits)   # 2.0 ATR — mai mult spatiu
        tp1_sr = nearest_zone(h4_sr.get('support', []), above=False)
        tp1    = tp1_sr if tp1_sr else round(entry - atr * 3.0, digits)
        tp2_sr = second_nearest_zone(h4_sr.get('support', []), above=False, first_level=tp1)
        tp2    = tp2_sr if tp2_sr else round(entry - atr * 5.0, digits)

    risk   = abs(entry - sl)
    reward = abs(tp1 - entry)
    rr     = round(reward / risk, 2) if risk > 0 else 0

    return {
        'entry_price':    round(entry, digits),
        'stop_loss':      round(sl, digits),
        'take_profit_1':  round(tp1, digits),
        'take_profit_2':  round(tp2, digits),
        'risk_pips':      round(risk * 10 ** digits, 1),
        'rr_ratio':       rr,
        'risk_percent':   0.75,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MOTIVE DECIZIE
# ═══════════════════════════════════════════════════════════════════════════════

def build_reasons(confluence, tf_analysis, direction, kill_zone):
    """Construieste o lista de motive pentru decizia de tranzactionare."""
    reasons = []

    kz_labels = {
        'london':  'Kill Zone Londra (07:00-10:00 UTC)',
        'newyork': 'Kill Zone New York (12:00-15:00 UTC)',
        'asia':    'Kill Zone Asia (00:00-03:00 UTC)'
    }
    if kill_zone in kz_labels:
        reasons.append(kz_labels[kill_zone])

    for zone in confluence.get('confluence_zones', [])[:4]:
        reasons.append(
            f"{zone['tf']} zona {'support' if zone['type']=='support' else 'rezistenta'} "
            f"la {zone['level']:.5f} (testat de {zone['strength']}x)"
        )

    for tf, det in confluence.get('details', {}).items():
        if det.get('fib'):
            reasons.append(f"{tf}: pret in zona OTE Fibonacci (61.8%-78.6%)")
        if det.get('ob'):
            reasons.append(f"{tf}: Order Block activ la pretul curent")
        if det.get('fvg'):
            reasons.append(f"{tf}: Fair Value Gap mitigat")

    h4_trend = tf_analysis.get('H4', {}).get('trend', '')
    d1_trend = tf_analysis.get('D1', {}).get('trend', '')
    if h4_trend and h4_trend == d1_trend:
        reasons.append(f"Aliniere trend: {h4_trend} pe H4 si D1")

    h1_rsi = tf_analysis.get('H1', {}).get('rsi', 50)
    if direction == 'BUY'  and h1_rsi < 38:
        reasons.append(f"H1 RSI supravandut la {h1_rsi:.1f}")
    elif direction == 'SELL' and h1_rsi > 62:
        reasons.append(f"H1 RSI supracumparat la {h1_rsi:.1f}")

    for tf in ['M5', 'M15']:
        mss = tf_analysis.get(tf, {}).get('mss')
        if mss:
            reasons.append(f"{tf} Market Structure Shift {mss}")
        disp = tf_analysis.get(tf, {}).get('displacement')
        if disp:
            reasons.append(f"{tf} displacement candle {disp}")

    return reasons[:8]


# ═══════════════════════════════════════════════════════════════════════════════
# STIRI FOREX (optional, refolosit din proiectul anterior)
# ═══════════════════════════════════════════════════════════════════════════════

def get_news_blackout_currencies():
    """Returneaza lista de valute afectate de stiri HIGH impact in urmatoarele 30 minute."""
    global _news_cache
    now = datetime.now(timezone.utc)

    # Cache de 5 minute
    if _news_cache['fetched_at'] and (now - _news_cache['fetched_at']).seconds < 300:
        news = _news_cache['data']
    else:
        try:
            resp = requests.get(
                'https://nfs.faireconomy.media/ff_calendar_thisweek.xml',
                timeout=5
            )
            root = ET.fromstring(resp.content)
            news = []
            for item in root.findall('event'):
                impact = item.findtext('impact', '')
                if impact.upper() != 'HIGH':
                    continue
                currency = item.findtext('country', '')
                date_str = item.findtext('date', '')
                time_str = item.findtext('time', '')
                try:
                    event_dt = datetime.strptime(
                        f"{date_str} {time_str}", "%m-%d-%Y %I:%M%p"
                    ).replace(tzinfo=timezone.utc)
                    news.append({'currency': currency, 'time': event_dt})
                except Exception:
                    pass
            _news_cache = {'data': news, 'fetched_at': now}
        except Exception as e:
            logger.warning(f"Eroare stiri: {e}")
            return set()

    blackout = set()
    for event in news:
        diff = (event['time'] - now).total_seconds() / 60
        if -NEWS_BLACKOUT_POST <= diff <= NEWS_BLACKOUT_PRE:
            blackout.add(event['currency'])
    return blackout


def is_affected_by_news(symbol, blackout_currencies):
    """Verifica daca un simbol este afectat de stiri HIGH impact."""
    sym_upper = symbol.upper()
    for currency in blackout_currencies:
        if currency in sym_upper:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/analyze_mtf', methods=['POST'])
def analyze_mtf():
    """Analiza multi-timeframe: primeste OHLCV, returneaza decizie + grafic."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'Date lipsa', 'decision': 'HOLD'}), 400

        symbol      = data.get('symbol', 'UNKNOWN')
        balance     = float(data.get('balance', 10000))
        equity      = float(data.get('equity', 10000))
        open_trades = int(data.get('open_trades', 0))
        kill_zone   = data.get('kill_zone', 'NONE')
        price       = float(data.get('current_price', 0))
        digits      = int(data.get('digits', 5))

        logger.info(f"Analiz {symbol} | Pret: {price} | Kill zone: {kill_zone}")

        def log_hold(sym, reason_text):
            _signal_log.append({
                'time':       datetime.now().strftime('%H:%M:%S'),
                'symbol':     sym,
                'decision':   'HOLD',
                'confidence': 0,
                'confluence': 0,
                'rr':         0,
                'entry':      0,
                'sl':         0,
                'tp1':        0,
                'kill_zone':  kill_zone,
                'reason':     reason_text,
                'chart_url':  '',
            })
            if len(_signal_log) > MAX_LOG_SIZE:
                _signal_log.pop(0)

        if price <= 0:
            return jsonify({'decision': 'HOLD', 'reason': 'Pret invalid'}), 200

        # ── Verifica stiri ──
        blackout = get_news_blackout_currencies()
        if is_affected_by_news(symbol, blackout):
            logger.info(f"{symbol}: Blocat de stiri HIGH impact")
            return jsonify({'decision': 'HOLD', 'reason': 'Stiri HIGH impact active'}), 200

        # ── Parseaza barele per timeframe ──
        tf_bars = data.get('timeframes', {})
        tf_dfs  = {}
        for tf_name, bars in tf_bars.items():
            df = bars_to_df(bars)
            if not df.empty:
                tf_dfs[tf_name] = df

        if not tf_dfs:
            return jsonify({'decision': 'HOLD', 'reason': 'Date OHLCV lipsa'}), 200

        # ── Swing-uri din 1000 bare (pentru Fibonacci precis) ──
        fib_swings = data.get('fib_swings', {})

        # ── Analiza per timeframe (cu fibonacci din 1000 bare) ──
        tf_analysis = {
            tf: analyze_tf(df, fib_swing=fib_swings.get(tf))
            for tf, df in tf_dfs.items()
        }

        # ── Confluenta ──
        confluence      = check_confluence(price, tf_analysis, digits)
        direction       = confluence['direction']
        conf_score      = confluence['score']
        conf_score_max  = confluence['max_score']
        dir_score       = confluence.get('direction_score', conf_score)

        logger.info(f"{symbol}: {direction} | Scor confluenta: {conf_score}/{conf_score_max} | Dir: {dir_score}")

        # ── Verifica scor minim ──
        if direction == 'HOLD' or conf_score < MIN_CONFLUENCE:
            r = f'Confluenta insuficienta: {conf_score}/{conf_score_max}'
            log_hold(symbol, r)
            return jsonify({'decision': 'HOLD', 'confluence_score': conf_score, 'reason': r}), 200

        # ── Gate H4 trend: tranzactionam DOAR cu trendul H4 ──
        h4_ana   = tf_analysis.get('H4', {})
        h4_bias  = h4_ana.get('bias', '')
        h4_trend = h4_ana.get('trend', '')
        if direction == 'BUY' and h4_bias == 'bearish' and h4_trend == 'bearish':
            r = 'BUY contra trend H4 (EMA50+EMA200 bearish)'
            logger.info(f"{symbol}: {r}")
            log_hold(symbol, r)
            return jsonify({'decision': 'HOLD', 'reason': r, 'confluence_score': conf_score}), 200
        if direction == 'SELL' and h4_bias == 'bullish' and h4_trend == 'bullish':
            r = 'SELL contra trend H4 (EMA50+EMA200 bullish)'
            logger.info(f"{symbol}: {r}")
            log_hold(symbol, r)
            return jsonify({'decision': 'HOLD', 'reason': r, 'confluence_score': conf_score}), 200

        # ── Filtru RSI extrem ──
        h1_rsi = tf_analysis.get('H1', {}).get('rsi', 50)
        if direction == 'BUY' and h1_rsi > 68:
            r = f'BUY blocat — RSI H1 supracumparat ({h1_rsi:.0f})'
            logger.info(f"{symbol}: {r}")
            log_hold(symbol, r)
            return jsonify({'decision': 'HOLD', 'reason': r, 'confluence_score': conf_score}), 200
        if direction == 'SELL' and h1_rsi < 32:
            r = f'SELL blocat — RSI H1 supravandut ({h1_rsi:.0f})'
            logger.info(f"{symbol}: {r}")
            log_hold(symbol, r)
            return jsonify({'decision': 'HOLD', 'reason': r, 'confluence_score': conf_score}), 200

        # ── Nivelele trade-ului ──
        levels = calc_trade_levels(price, direction, tf_analysis, digits)

        if levels['rr_ratio'] < MIN_RR:
            r = f"R:R prea mic: {levels['rr_ratio']:.2f} < {MIN_RR}"
            log_hold(symbol, r)
            return jsonify({'decision': 'HOLD', 'reason': r, 'confluence_score': conf_score}), 200

        # ── Motive ──
        reasons = build_reasons(confluence, tf_analysis, direction, kill_zone)

        # ── Confidence score — bazat pe scorul DIRECTIONAL, nu total ──
        # Confidence = cat % din scorul activ e in directia aleasa
        # Exemplu: bull=20, bear=5 → 20/(20+5) = 80%
        active_total = confluence['bull_score'] + confluence['bear_score']
        confidence = min(int(dir_score / max(active_total, 1) * 100), 95)

        # ── Genereaza grafic ──
        chart_filename = ''
        chart_url      = ''
        try:
            # Pregateste datele de analiza pentru chart_generator
            chart_analysis = {
                'fibonacci':    {tf: tf_analysis[tf].get('fibonacci')    for tf in tf_analysis},
                'sr_zones':     {tf: tf_analysis[tf].get('sr_zones', {}) for tf in tf_analysis},
                'order_blocks': {tf: tf_analysis[tf].get('order_blocks', []) for tf in tf_analysis},
                'fvg':          {tf: tf_analysis[tf].get('fvg', [])      for tf in tf_analysis},
            }
            chart_decision = {
                **levels,
                'decision':        direction,
                'reasons':         reasons,
                'confidence_score': confidence,
                'confluence_score': conf_score,
                'symbol':          symbol,
            }
            chart_filename = create_trade_chart(
                symbol         = symbol,
                timeframe_data = tf_dfs,
                analysis       = chart_analysis,
                decision       = chart_decision,
                chart_dir      = CHARTS_DIR
            )
            chart_url = f"http://{SERVER_HOST}:{SERVER_PORT}/charts/{chart_filename}"
            logger.info(f"Grafic generat: {chart_filename}")

        except Exception as e:
            logger.error(f"Eroare generare grafic: {e}", exc_info=True)

        # ── Raspuns final ──
        response = {
            'decision':         direction,
            'symbol':           symbol,
            **levels,
            'confidence_score': confidence,
            'confluence_score': conf_score,
            'reasons':          reasons,
            'kill_zone':        kill_zone,
            'chart_path':       chart_filename,
            'chart_url':        chart_url,
        }

        logger.info(
            f"SEMNAL {symbol}: {direction} | "
            f"Entry: {levels['entry_price']} | SL: {levels['stop_loss']} | "
            f"TP1: {levels['take_profit_1']} | R:R 1:{levels['rr_ratio']} | "
            f"Confidence: {confidence}%"
        )

        # ── Inregistreaza in log pentru /signals dashboard ──
        _signal_log.append({
            'time':       datetime.now().strftime('%H:%M:%S'),
            'symbol':     symbol,
            'decision':   direction,
            'confidence': confidence,
            'confluence': conf_score,
            'rr':         levels.get('rr_ratio', 0),
            'entry':      levels.get('entry_price', 0),
            'sl':         levels.get('stop_loss', 0),
            'tp1':        levels.get('take_profit_1', 0),
            'kill_zone':  kill_zone,
            'reason':     reasons[0] if reasons else '',
            'chart_url':  chart_url,
        })
        if len(_signal_log) > MAX_LOG_SIZE:
            _signal_log.pop(0)

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Eroare server: {e}", exc_info=True)
        return jsonify({'error': str(e), 'decision': 'HOLD'}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# SERVIRE GRAFICE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/monitor_mtf', methods=['POST'])
def monitor_mtf():
    """
    Monitorizeaza pozitiile deschise si decide cand sa le inchida.

    Reguli de inchidere:
      1. Profit >= 2x risk_usd (target atins)
      2. Stire HIGH impact imminenta pe valuta simbolului
      3. Profit >= 100% din risk (safety lock)
      4. Pierdere aproape de SL (profit <= -45 din 50 risc)
    """
    try:
        data      = request.get_json(force=True)
        positions = data.get('positions', [])
        blackout  = get_news_blackout_currencies()
        decisions = {}

        for pos in positions:
            ticket   = str(pos.get('ticket', ''))
            symbol   = pos.get('symbol', '')
            pos_type = pos.get('type', '')
            entry    = float(pos.get('entry', 0))
            sl       = float(pos.get('sl', 0))
            profit   = float(pos.get('profit', 0))
            risk_usd = float(pos.get('risk_usd', 50))
            target   = risk_usd * 2  # ex: $100 profit tinta

            reason = None

            # Regula 1: profit tinta atins (2:1)
            if profit >= target:
                reason = f"Profit tinta atins: ${profit:.2f} >= ${target:.2f}"

            # Regula 2: stire HIGH impact
            elif is_affected_by_news(symbol, blackout):
                reason = f"Stire HIGH impact imminenta pe {symbol}"

            # Regula 3: aproape de stop loss
            elif profit <= -(risk_usd * 0.9):
                reason = f"Aproape de stop loss: ${profit:.2f}"

            if reason:
                decisions[ticket] = 'CLOSE'
                logger.info(f"CLOSE #{ticket} {symbol}: {reason}")
            else:
                decisions[ticket] = 'HOLD'

        return jsonify(decisions), 200

    except Exception as e:
        logger.error(f"Monitor error: {e}", exc_info=True)
        return jsonify({}), 500


@app.route('/review_trade', methods=['POST'])
def review_trade():
    """
    Re-analizeaza un trade deschis pe baza datelor curente de piata.

    Input JSON:
      ticket        - numarul ticketului MT4/MT5
      symbol        - ex: "EURUSD"
      direction     - "BUY" sau "SELL" (directia originala)
      entry_price   - pretul de intrare
      stop_loss     - SL curent
      take_profit_1 - TP1 curent
      current_price - pretul curent al pietei
      digits        - numarul de zecimale (ex: 5)
      timeframes    - dict cu bare OHLCV per timeframe (same format ca /analyze_mtf)

    Output JSON:
      ticket        - ticketul primit
      still_valid   - true/false
      action        - "HOLD" (pastreaza) sau "CLOSE" (inchide)
      reason        - explicatie
      confluence_score / max_score - scorul curent de confluenta
      current_direction - directia pe care o vede piata acum
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'Date lipsa'}), 400

        ticket        = str(data.get('ticket', '?'))
        symbol        = data.get('symbol', 'UNKNOWN')
        orig_direction = data.get('direction', '').upper()
        entry_price   = float(data.get('entry_price', 0))
        sl            = float(data.get('stop_loss', 0))
        tp1           = float(data.get('take_profit_1', 0))
        price         = float(data.get('current_price', 0))
        digits        = int(data.get('digits', 5))

        if price <= 0 or orig_direction not in ('BUY', 'SELL'):
            return jsonify({
                'ticket': ticket,
                'still_valid': False,
                'action': 'HOLD',
                'reason': 'Date de intrare invalide (pret sau directie lipsa)'
            }), 200

        logger.info(f"Review #{ticket} {symbol} {orig_direction} | Entry: {entry_price} | Pret curent: {price}")

        # ── Verifica stiri HIGH impact ──
        blackout = get_news_blackout_currencies()
        if is_affected_by_news(symbol, blackout):
            return jsonify({
                'ticket': ticket,
                'still_valid': False,
                'action': 'CLOSE',
                'reason': 'Stire HIGH impact imminenta — inchide pozitia'
            }), 200

        # ── Parseaza barele curente ──
        tf_bars = data.get('timeframes', {})
        tf_dfs = {}
        for tf_name, bars in tf_bars.items():
            df = bars_to_df(bars)
            if not df.empty:
                tf_dfs[tf_name] = df

        if not tf_dfs:
            return jsonify({
                'ticket': ticket,
                'still_valid': True,
                'action': 'HOLD',
                'reason': 'Date OHLCV lipsa — nu pot evalua, pastrez pozitia'
            }), 200

        # ── Re-analizeaza piata actuala ──
        tf_analysis = {tf: analyze_tf(df) for tf, df in tf_dfs.items()}
        confluence  = check_confluence(price, tf_analysis, digits)

        current_direction = confluence['direction']   # BUY / SELL / HOLD
        conf_score        = confluence['score']
        conf_score_max    = confluence['max_score']

        reasons = []

        # ── Regula 1: directia pietei s-a inversat ──
        if current_direction not in ('HOLD', orig_direction):
            reasons.append(
                f"Directia pietei s-a inversat: piata arata {current_direction}, "
                f"trade-ul este {orig_direction}"
            )

        # ── Regula 2: confluenta a scazut sub minim ──
        if conf_score < MIN_CONFLUENCE:
            reasons.append(
                f"Confluenta actuala insuficienta: {conf_score}/{conf_score_max} "
                f"(min: {MIN_CONFLUENCE})"
            )

        # ── Regula 3: pretul a depasit TP1 (trade posibil epuizat) ──
        if tp1 > 0:
            if orig_direction == 'BUY' and price >= tp1:
                reasons.append(f"Pretul {price} a atins/depasit TP1 {tp1} — luati profitul")
            elif orig_direction == 'SELL' and price <= tp1:
                reasons.append(f"Pretul {price} a atins/depasit TP1 {tp1} — luati profitul")

        # ── Decizie finala ──
        if reasons:
            action      = 'CLOSE'
            still_valid = False
            reason_text = ' | '.join(reasons)
        else:
            action      = 'HOLD'
            still_valid = True
            reason_text = (
                f"Setup intact: {orig_direction} confirmat cu confluenta "
                f"{conf_score}/{conf_score_max}"
            )

        logger.info(f"Review #{ticket}: {action} — {reason_text}")

        return jsonify({
            'ticket':            ticket,
            'symbol':            symbol,
            'still_valid':       still_valid,
            'action':            action,
            'reason':            reason_text,
            'original_direction': orig_direction,
            'current_direction': current_direction,
            'confluence_score':  conf_score,
            'max_score':         conf_score_max,
        }), 200

    except Exception as e:
        logger.error(f"Review trade error: {e}", exc_info=True)
        return jsonify({'error': str(e), 'action': 'HOLD'}), 500


@app.route('/charts/<filename>')
def serve_chart(filename):
    """Serveste un grafic PNG."""
    return send_from_directory(CHARTS_DIR, filename)


@app.route('/charts')
def charts_dashboard():
    """Dashboard HTML cu toate graficele recente."""
    charts = []
    if os.path.exists(CHARTS_DIR):
        charts = sorted(
            [f for f in os.listdir(CHARTS_DIR) if f.endswith('.png')],
            reverse=True
        )[:30]

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f"""<!DOCTYPE html>
<html lang="ro">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="refresh" content="60">
  <title>MultiTFTrader — Grafice</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #0d1117; color: #e6edf3; font-family: 'Courier New', monospace; padding: 24px; }}
    h1 {{ color: #58a6ff; font-size: 1.6rem; margin-bottom: 6px; }}
    .meta {{ color: #8b949e; font-size: 0.82rem; margin-bottom: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 18px; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px;
             padding: 12px; transition: border-color 0.2s; }}
    .card:hover {{ border-color: #58a6ff; }}
    .card img {{ width: 100%; border-radius: 6px; cursor: pointer; display: block; }}
    .card-name {{ color: #8b949e; font-size: 0.75rem; margin-top: 8px; text-align: center; }}
    .card-name a {{ color: #58a6ff; text-decoration: none; }}
    .card-name a:hover {{ text-decoration: underline; }}
    .buy {{ border-left: 3px solid #26a641; }}
    .sell {{ border-left: 3px solid #f85149; }}
    .empty {{ color: #8b949e; text-align: center; padding: 60px; font-size: 1.1rem; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
              font-size: 0.72rem; font-weight: bold; margin-left: 8px; }}
    .badge-buy  {{ background: #26a641; color: #0d1117; }}
    .badge-sell {{ background: #f85149; color: #0d1117; }}
  </style>
</head>
<body>
  <h1>MultiTFTrader <span style="color:#8b949e;font-size:1rem">/ Grafice Decizii</span></h1>
  <div class="meta">Actualizat: {now_str} &nbsp;|&nbsp; Refresh automat la 60s &nbsp;|&nbsp;
    <a href="/health" style="color:#58a6ff">Status server</a></div>
  <div class="grid">
"""

    if not charts:
        html += '<div class="empty">Niciun grafic inca. Asteptam semnale...</div>'
    else:
        for fname in charts:
            parts = fname.replace('.png', '').split('_')
            sym   = parts[0] if parts else ''
            act   = parts[1] if len(parts) > 1 else ''
            cls   = 'buy' if act == 'BUY' else ('sell' if act == 'SELL' else '')
            badge = f'<span class="badge badge-{act.lower()}">{act}</span>' if act in ('BUY', 'SELL') else ''

            html += f"""
    <div class="card {cls}">
      <a href="/charts/{fname}" target="_blank">
        <img src="/charts/{fname}" alt="{fname}" loading="lazy">
      </a>
      <div class="card-name">
        <strong>{sym}</strong>{badge} &nbsp;
        <a href="/charts/{fname}" target="_blank">{fname}</a>
      </div>
    </div>
"""

    html += """
  </div>
</body>
</html>"""

    return html


@app.route('/signals')
def signals_dashboard():
    """Dashboard live cu ultimele semnale generate."""
    rows = ''
    for s in reversed(_signal_log):
        dec = s['decision']
        if dec == 'BUY':
            color = '#238636'; badge = '▲ BUY'
        elif dec == 'SELL':
            color = '#da3633'; badge = '▼ SELL'
        else:
            color = '#30363d'; badge = '— HOLD'

        chart_link = (f'<a href="{s["chart_url"]}" target="_blank">grafic</a>'
                      if s.get('chart_url') else '')
        entry_info = (f"{s['entry']} / SL {s['sl']} / TP {s['tp1']} / R:R {s['rr']}"
                      if dec in ('BUY','SELL') else '')

        rows += f"""
        <tr>
          <td style="color:#8b949e">{s['time']}</td>
          <td><b>{s['symbol']}</b></td>
          <td><span style="background:{color};padding:2px 8px;border-radius:4px;font-weight:bold">{badge}</span></td>
          <td>{s['confidence']}%</td>
          <td>{s['confluence']}</td>
          <td style="color:#8b949e;font-size:.8rem">{s['reason'][:60]}</td>
          <td style="color:#8b949e;font-size:.8rem">{entry_info}</td>
          <td>{chart_link}</td>
        </tr>"""

    total   = len(_signal_log)
    trades  = sum(1 for s in _signal_log if s['decision'] in ('BUY','SELL'))
    holds   = total - trades

    return f"""<!DOCTYPE html>
<html lang="ro"><head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="15">
<title>MultiTFTrader — Semnale Live</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;color:#e6edf3;font-family:'Courier New',monospace;padding:24px}}
h1{{color:#58a6ff;margin-bottom:6px}}
.stats{{color:#8b949e;font-size:.85rem;margin-bottom:20px}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{background:#161b22;color:#8b949e;padding:8px 10px;text-align:left;border-bottom:1px solid #30363d}}
td{{padding:7px 10px;border-bottom:1px solid #21262d}}
tr:hover td{{background:#161b22}}
a{{color:#58a6ff}}
</style></head><body>
<h1>MultiTFTrader — Semnale Live</h1>
<div class="stats">
  Ultimele {total} semnale &nbsp;|&nbsp;
  <span style="color:#238636">{trades} trades</span> &nbsp;|&nbsp;
  {holds} HOLD &nbsp;|&nbsp;
  Refresh automat la 15s &nbsp;|&nbsp;
  <span style="color:#8b949e">{datetime.now().strftime('%H:%M:%S')}</span>
</div>
<table>
  <tr>
    <th>Ora</th><th>Simbol</th><th>Decizie</th>
    <th>Conf%</th><th>Score</th><th>Motiv</th>
    <th>Niveluri</th><th>Chart</th>
  </tr>
  {rows if rows else '<tr><td colspan="8" style="color:#8b949e;text-align:center;padding:30px">Niciun semnal inca — EA trimite date la fiecare 60s</td></tr>'}
</table>
</body></html>"""


@app.route('/health')
def health():
    """Status server."""
    return jsonify({
        'status':    'OK',
        'server':    'MultiTFTrader v1.0',
        'charts':    len([f for f in os.listdir(CHARTS_DIR) if f.endswith('.png')])
                     if os.path.exists(CHARTS_DIR) else 0,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 200


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info("=" * 55)
    logger.info("  MultiTFTrader Server v1.0 pornit")
    logger.info(f"  Dashboard: http://{SERVER_HOST}:{SERVER_PORT}/charts")
    logger.info(f"  Health:    http://{SERVER_HOST}:{SERVER_PORT}/health")
    logger.info("=" * 55)
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)
