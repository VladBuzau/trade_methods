# MultiTFTrader — Multi-Timeframe ICT/SMC cu Fibonacci & Zone Support

## Ce face acest sistem

Detecteaza zone de support si rezistenta pe **5 timeframe-uri** (M1, M5, H1, H4, D1),
verifica **confluenta** intre ele si ia decizii de tranzactionare bazate pe:

| Concept | Detalii |
|---|---|
| **Zone S/R** | Clustering de swing highs/lows per TF |
| **Fibonacci OTE** | 61.8% si 78.6% (Optimal Trade Entry) |
| **Order Blocks** | Ultima lumanare opusa inainte de miscare impulsiva |
| **Fair Value Gaps** | Goluri de pret neacoperite |
| **Kill Zones** | Londra 07-10 UTC, New York 12-15 UTC |
| **Risk** | 0.5%-1% per trade, R:R minim 1:2 |

Pentru **fiecare semnal** se genereaza automat un **grafic PNG** cu:
- Candlestick-uri pe H4, D1, H1, M5
- Linii Fibonacci (23.6%, 38.2%, 50%, 61.8%, 78.6%)
- Zone S/R colorate (verde = support, rosu = rezistenta)
- Order Blocks si Fair Value Gaps vizualizate
- Entry / Stop Loss / TP1 / TP2 marcate
- Motivele deciziei in text

---

## Structura proiect

```
MultiTFTrader/
├── EA/
│   └── MultiTFTrader.mq5      ← EA MetaTrader 5
└── server/
    ├── server.py               ← Server Flask (analiza + grafice)
    ├── chart_generator.py      ← Generare grafice matplotlib
    ├── config.json             ← Configurare
    └── requirements.txt        ← Dependinte Python
```

---

## Instalare si pornire

### 1. Server Python

```bash
cd MultiTFTrader/server
pip install -r requirements.txt
python server.py
```

Serverul porneste pe `http://localhost:5001`.

**Dashboard grafice:** `http://localhost:5001/charts`

---

### 2. EA MetaTrader 5

1. Copiaza `MultiTFTrader.mq5` in `MQL5/Experts/`
2. Compileaza in MetaEditor (F7)
3. **Permite WebRequest:**
   - MT5 → Tools → Options → Expert Advisors
   - Bifeaza "Allow WebRequest for listed URL"
   - Adauga: `http://localhost:5001`
4. Ataseaza EA pe orice grafic (timeframe-ul nu conteaza, EA foloseste toate TF-urile)
5. Seteaza parametrii in panoul EA

---

## Parametri EA

| Parametru | Default | Descriere |
|---|---|---|
| InpServerURL | `http://localhost:5001` | Adresa serverului |
| InpRiskPercent | `0.75` | Risc % per trade |
| InpMinRR | `2.0` | R:R minim |
| InpMaxOpenTrades | `2` | Max pozitii simultane |
| InpDailyLossLimit | `4.0` | Limita pierdere zilnica % |
| InpMagicNumber | `77777` | Magic number |
| InpScanInterval | `60` | Interval scanare (secunde) |
| InpBarsPerTF | `60` | Bare per timeframe trimise |
| InpUseKillZones | `true` | Restrictie kill zone |
| InpMinConfidence | `60` | Incredere minima (%) |
| InpMinConfluence | `5` | Scor confluenta minim |
| InpSymbols | `EURUSD,...` | Simboluri de analizat |

---

## Logica de confluenta

Fiecare confirmare pe un timeframe adauga puncte la scor (timeframe mai mare = mai mult):

| Timeframe | Pondere | Confirmare posibila |
|---|---|---|
| M1 | 1 | Zona S/R, Fibonacci OTE, OB |
| M5 | 2 | Zona S/R, Fibonacci OTE, OB |
| H1 | 3 | Zona S/R, Fibonacci OTE, OB |
| H4 | 4 | Zona S/R, Fibonacci OTE, OB |
| D1 | 5 | Zona S/R, Fibonacci OTE, OB |

**Exemplu:** semnal BUY valid:
- H1 zona support + Fibonacci 61.8% → 3+3 = 6 puncte
- H4 Order Block bullish → +4 = 10 puncte
- D1 zona support → +5 = 15 puncte ✅

Pragul minim implicit: **5 puncte** (configurabil in `config.json`).

---

## Graficele generate

Fiecare semnal produce un PNG cu 4 panouri:

```
┌─────────────────────────────┬──────────────┐
│  H4 — Bias / Trend           │  D1 — Zilnic │
│  (zone, OB, FVG, Fib)        │              │
├─────────────────────────────┼──────────────┤
│  H1 — Intrare                │  M5 — Confirm│
│  (Entry, SL, TP1, TP2)       │              │
└─────────────────────────────┴──────────────┘
```

Graficele se pot vedea la: `http://localhost:5001/charts`
