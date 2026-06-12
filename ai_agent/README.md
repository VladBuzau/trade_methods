# AI Agent — Trading ML pipeline

Sistem de trading bazat pe ML. Combina pret istoric + stiri + macro pentru a
genera semnale de tranzactionare. Integreaza in ChartVisualizer ca strategie noua.

## Statut faze

- [x] **Faza 0** — Plan + arhitectura
- [x] **Faza 1** — Data pipeline (acest commit)
  - [x] SQLite schema
  - [x] MT5 historical dumper
  - [x] News collector (AlphaVantage + NewsAPI + Reddit)
  - [x] Macro collector (FRED)
- [ ] **Faza 2** — Feature engineering
- [ ] **Faza 3** — Sentiment NLP (FinBERT)
- [ ] **Faza 4** — Model training (XGBoost + LSTM)
- [ ] **Faza 5** — Backtest + paper trade
- [ ] **Faza 6** — Production + monitoring

## Setup

```bash
# 1. Instaleaza dependintele
cd ai_agent
pip install -r requirements.txt

# 2. Initializeaza DB
python -m ai_agent.db.schema

# 3. Seteaza API keys in env (sau .env file)
#    Toate sunt OPTIONALE — pipeline-ul merge si fara, dar features lipsesc
export ALPHA_VANTAGE_KEY=your_key_here    # https://www.alphavantage.co/support/#api-key
export NEWSAPI_KEY=your_key_here          # https://newsapi.org
export FRED_KEY=your_key_here             # https://fred.stlouisfed.org/docs/api/api_key.html
export REDDIT_CLIENT_ID=your_id           # https://www.reddit.com/prefs/apps
export REDDIT_CLIENT_SECRET=your_secret

# 4. Verifica setarile
python -m ai_agent.config
```

## Comenzi Faza 1

```bash
# Descarca tot istoricul MT5 (5 ani × 11 simboluri × 5 TF = ~3-6h pe primul run)
python -m ai_agent.data.mt5_dumper

# Sau doar un simbol
python -m ai_agent.data.mt5_dumper EURUSD H1

# News:
python -m ai_agent.data.news_collector alphavantage EURUSD
python -m ai_agent.data.news_collector newsapi "USD inflation"
python -m ai_agent.data.news_collector reddit forex 50
python -m ai_agent.data.news_collector all_reddit 30

# Macro (FRED):
python -m ai_agent.data.macro_collector

# Stats DB
python -m ai_agent.db.schema
```

## API Keys — cum le obtii

| Sursa | Cost | Link |
|---|---|---|
| **AlphaVantage** (news + sentiment) | FREE 25/zi · $50/mo unlimited | https://www.alphavantage.co/support/#api-key |
| **NewsAPI** (news general) | FREE 100/zi dev · $449/mo prod | https://newsapi.org |
| **FRED** (macro: yields, VIX, M2) | FREE (cu inregistrare) | https://fred.stlouisfed.org/docs/api/api_key.html |
| **Reddit** (forex/crypto sentiment) | FREE | https://www.reddit.com/prefs/apps |
| **Twitter/X** (Trump, Fed officials) | $100/mo basic | https://developer.twitter.com |

## Structura proiect

```
ai_agent/
├── config.py              # config centralizat + env keys
├── db/
│   ├── schema.py          # SQLite tables init
│   └── agent.db           # database (auto-creat)
├── data/                  # FAZA 1 — collectors
│   ├── mt5_dumper.py
│   ├── news_collector.py
│   └── macro_collector.py
├── features/              # FAZA 2 — feature engineering (TODO)
├── models/                # FAZA 3-4 — sentiment + training (TODO)
├── strategies_ai/         # FAZA 5 — strategie compatibila ChartVisualizer (TODO)
└── logs/
```

## Filozofie

1. **Decision augmentation, NU autopilot** — modelele dau scoruri de confidence,
   tu (sau scanner-ul existent) decizi
2. **Walk-forward training obligatoriu** — niciodata random split
3. **Out-of-sample validation min 3 luni** inainte de paper trade
4. **Paper trade min 2-4 sapt** inainte de capital real
5. **Kill switch automat** — orice DD > 10% opreste agentul

## Honest disclaimers

- Majoritatea sistemelor ML pe trading **pierd in productie** desi backtest-urile
  arata bine (overfitting, regime change, slippage real).
- News alpha (Trump tweets etc.) e **mostly arbitraged** de HFT-uri din 2020.
- Acest sistem are sens ca **decision augmentation** + **structurare disciplinata**,
  nu ca cale rapida spre profit.
- Backtest-uri impressive ≠ profit live.
