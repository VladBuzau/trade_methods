"""
config.py — constante imutabile pentru ChartVisualizer/AutoTrader.

Aici sunt centralizate listele/dicts care nu se modifica runtime:
  - Symbol-uri tranzactionate (forex + crypto)
  - Timeframes MT5 + bars per TF
  - Spread limits
  - Prioritati strategii
  - Constante FTMO / performance

Variabilele care SE modifica runtime (RISK_DOLLARS, MAX_OPEN_TRADES, etc.)
raman in app.py pentru a permite re-binding via `_app.RISK_DOLLARS = X`.
"""
from __future__ import annotations

# ── Timeframes ──────────────────────────────────────────────────────────────
MT5_TF: dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15,
    "M30": 30, "H1": 16385, "H4": 16388, "D1": 16408,
}
ALL_TFS: list[str] = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
MULTI_BARS: dict[str, int] = {t: 2000 for t in ALL_TFS}


# ── Symbol-uri ──────────────────────────────────────────────────────────────
SYMBOLS: list[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "XAUUSD",
    "USDCHF",
]

SYMBOLS_CRYPTO: list[str] = [
    # Top volum & lichiditate
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    "XRPUSD",   # Ripple
    "SOLUSD",   # Solana
    "BNBUSD",   # BNB
    "DOGEUSD",  # Dogecoin
    "ADAUSD",   # Cardano
    "AVAXUSD",  # Avalanche
    "LINKUSD",  # Chainlink
    "LTCUSD",   # Litecoin
]


# ── Spread limits ───────────────────────────────────────────────────────────
MAX_SPREAD_PIPS: dict[str, float] = {
    # Forex majors
    "EURUSD": 1.5, "GBPUSD": 1.8, "USDJPY": 1.5, "USDCHF": 1.8,
    "USDCAD": 2.0, "AUDUSD": 1.8, "NZDUSD": 2.0, "EURGBP": 1.8,
    "GBPJPY": 3.0, "EURJPY": 2.5, "AUDJPY": 3.0,
    # Metale
    "XAUUSD": 5.0, "XAGUSD": 8.0,
    # Indici
    "US30":   200.0, "US500":  150.0, "US100":  150.0,
    "NAS100": 150.0, "SPX500": 150.0, "UK100":  150.0,
    "GER40":  200.0, "GER30":  200.0, "FRA40":  200.0,
    "AUS200": 150.0, "JPN225": 300.0,
    # Crypto
    "BTCUSD": 5000.0, "ETHUSD": 200.0,
}
MAX_SPREAD_DEFAULT: float = 100.0


# ── Prioritati strategii (P4) ───────────────────────────────────────────────
STRATEGY_PRIORITY: dict[str, int] = {
    "eob":              1,
    "smc":              2,
    "classic":          3,
    "london_breakout":  4,
    "ny_breakout":      4,
    "china_session":    5,
    "vwap_bounce":      5,
    "macd":             6,
    "supertrend":       6,
    "keltner_channel":  6,
    "rsi_divergence":   7,
    "bollinger":        7,
    "engulfing":        8,
    "ichimoku":         8,
    "ema_cross":        9,
}


# ── ADX / news / performance ────────────────────────────────────────────────
ADX_MIN: float = 20.0
FTMO_NEWS_BLOCK_MIN: int = 2

PERF_MIN_TRADES: int = 20
PERF_LOOKBACK: int = 50
PERF_MIN_PF: float = 1.0
