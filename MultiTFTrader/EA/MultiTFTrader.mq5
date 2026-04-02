//+------------------------------------------------------------------+
//|                                            MultiTFTrader.mq5     |
//|     Multi-Timeframe ICT/SMC Support Zone & Fibonacci Trader      |
//+------------------------------------------------------------------+
#property copyright "MultiTF Trader"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//--- Input Groups
input group "=== Server ==="
input string   InpServerURL      = "https://postlarval-barb-delineative.ngrok-free.dev";

input group "=== Risk Management ==="
input double   InpMaxRiskUSD     = 50.0;    // Risk maxim in USD per trade ($50)
input double   InpMinRR          = 2.0;     // R:R minim (profit = 2x risk)
input int      InpMaxOpenTrades  = 5;       // Max 5 pozitii simultane
input double   InpDailyLossLimit = 4.0;     // Daily loss limit %

input group "=== Strategy ==="
input int      InpMagicNumber    = 77777;
input int      InpScanInterval   = 60;      // Scanare la 60 secunde
input int      InpBarsPerTF      = 200;     // Bare pentru analiza/grafic
input int      InpFibLookback    = 1000;    // Bare pentru calcul Fibonacci
input bool     InpUseKillZones   = true;
input int      InpMinConfidence  = 65;      // Precizie minima 65%
input int      InpReviewInterval = 15;      // Review pozitii deschise la fiecare N minute

input group "=== Simboluri ==="
input string   InpSymbols = "EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD,USDCAD,NZDUSD,EURGBP,EURJPY,GBPJPY,AUDJPY,EURAUD,GBPAUD,XAUUSD,XAGUSD";

//--- Globals
CTrade         g_trade;
CPositionInfo  g_position;
string         g_symbols[];
int            g_symbol_count;
double         g_day_start_bal  = 0;
datetime       g_last_scan      = 0;
int            g_monitor_tick   = 0;
int            g_review_tick    = 0;

//+------------------------------------------------------------------+
int OnInit()
{
    g_trade.SetExpertMagicNumber(InpMagicNumber);
    g_trade.SetDeviationInPoints(50);
    g_trade.SetTypeFilling(ORDER_FILLING_IOC);

    g_symbol_count  = StringSplit(InpSymbols, ',', g_symbols);
    g_day_start_bal = AccountInfoDouble(ACCOUNT_BALANCE);

    EventSetTimer(InpScanInterval);

    // Buton Review manual
    ObjectCreate(0, "btn_review", OBJ_BUTTON, 0, 0, 0);
    ObjectSetInteger(0, "btn_review", OBJPROP_XDISTANCE, 10);
    ObjectSetInteger(0, "btn_review", OBJPROP_YDISTANCE, 30);
    ObjectSetInteger(0, "btn_review", OBJPROP_XSIZE, 140);
    ObjectSetInteger(0, "btn_review", OBJPROP_YSIZE, 28);
    ObjectSetString (0, "btn_review", OBJPROP_TEXT, "🔍 Review Trades");
    ObjectSetInteger(0, "btn_review", OBJPROP_COLOR, clrWhite);
    ObjectSetInteger(0, "btn_review", OBJPROP_BGCOLOR, clrDarkSlateGray);
    ObjectSetInteger(0, "btn_review", OBJPROP_BORDER_COLOR, clrSilver);
    ObjectSetInteger(0, "btn_review", OBJPROP_FONTSIZE, 9);
    ObjectSetInteger(0, "btn_review", OBJPROP_CORNER, CORNER_LEFT_UPPER);

    Print("╔══════════════════════════════════════════╗");
    Print("║   MultiTFTrader v2.00 Pornit             ║");
    Print("╚══════════════════════════════════════════╝");
    Print("Risk maxim: $", InpMaxRiskUSD, " per trade");
    Print("Max ", InpMaxOpenTrades, " pozitii simultane");
    Print("Fibonacci pe ", InpFibLookback, " candele");
    Print("Precizie minima: ", InpMinConfidence, "%");
    Print("Server: ", InpServerURL);
    return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
    EventKillTimer();
    ObjectDelete(0, "btn_review");
}

void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
{
    if(id == CHARTEVENT_OBJECT_CLICK && sparam == "btn_review")
    {
        ObjectSetInteger(0, "btn_review", OBJPROP_STATE, false);
        Print("[Review] Declansat manual...");
        ReviewOpenPositions();
    }
}

void OnTimer()
{
    // Monitorizeaza pozitiile deschise la fiecare apel (profit/loss/stiri)
    MonitorOpenPositions();

    // Review tehnic al pozitiilor deschise la fiecare InpReviewInterval minute
    g_review_tick++;
    if(g_review_tick >= InpReviewInterval)
    {
        g_review_tick = 0;
        ReviewOpenPositions();
    }

    // Break-even management la fiecare tick de timer
    ManageBreakEven();

    // Cauta semnale noi la fiecare 60 secunde
    g_monitor_tick++;
    if(g_monitor_tick >= 1)
    {
        g_monitor_tick = 0;
        ScanAllSymbols();
    }
}

void OnTick() {}

//+------------------------------------------------------------------+
string GetKillZone()
{
    datetime utc = TimeGMT();
    MqlDateTime dt;
    TimeToStruct(utc, dt);
    int hm = dt.hour * 60 + dt.min;
    if(hm >= 420  && hm < 600)  return "london";
    if(hm >= 720  && hm < 900)  return "newyork";
    if(hm >= 0    && hm < 180)  return "asia";
    return "NONE";
}

int CountOpenTrades()
{
    int count = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
        if(g_position.SelectByIndex(i) && g_position.Magic() == InpMagicNumber)
            count++;
    return count;
}

bool HasOpenPosition(string symbol)
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
        if(g_position.SelectByIndex(i))
            if(g_position.Symbol() == symbol && g_position.Magic() == InpMagicNumber)
                return true;
    return false;
}

// Numara cate pozitii deschise contin o anumita valuta (ex: "USD", "EUR")
int CountCurrencyExposure(string currency)
{
    int count = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(!g_position.SelectByIndex(i)) continue;
        if(g_position.Magic() != InpMagicNumber) continue;
        string sym = g_position.Symbol();
        if(StringFind(sym, currency) >= 0)
            count++;
    }
    return count;
}

bool IsDailyLossLimitReached()
{
    double equity   = AccountInfoDouble(ACCOUNT_EQUITY);
    double loss_pct = (g_day_start_bal - equity) / g_day_start_bal * 100.0;
    if(loss_pct >= InpDailyLossLimit)
    {
        Print("[!] Daily loss limit atins: ", DoubleToString(loss_pct, 2), "%");
        return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//| Returneaza bara ca JSON array [time,o,h,l,c,v]                   |
//+------------------------------------------------------------------+
string BarsToJSON(string symbol, ENUM_TIMEFRAMES tf, int count)
{
    MqlRates rates[];
    int copied = CopyRates(symbol, tf, 1, count, rates);
    if(copied <= 0) return "[]";

    string r = "[";
    for(int i = 0; i < copied; i++)
    {
        if(i > 0) r += ",";
        r += "[" + IntegerToString((long)rates[i].time)   + ","
               + DoubleToString(rates[i].open,  8) + ","
               + DoubleToString(rates[i].high,  8) + ","
               + DoubleToString(rates[i].low,   8) + ","
               + DoubleToString(rates[i].close, 8) + ","
               + IntegerToString(rates[i].tick_volume) + "]";
    }
    return r + "]";
}

//+------------------------------------------------------------------+
//| Calculeaza swing High/Low din ultimele N bare (pentru Fibonacci) |
//+------------------------------------------------------------------+
string GetFibSwing(string symbol, ENUM_TIMEFRAMES tf, int lookback)
{
    MqlRates rates[];
    int copied = CopyRates(symbol, tf, 1, lookback, rates);
    if(copied <= 0) return "{\"high\":0,\"low\":0,\"high_idx\":0,\"low_idx\":0}";

    double highest = rates[0].high;
    double lowest  = rates[0].low;
    int    hi_idx  = 0;
    int    lo_idx  = 0;

    for(int i = 1; i < copied; i++)
    {
        if(rates[i].high > highest) { highest = rates[i].high; hi_idx = i; }
        if(rates[i].low  < lowest)  { lowest  = rates[i].low;  lo_idx = i; }
    }

    return "{\"high\":"     + DoubleToString(highest, 8) +
           ",\"low\":"      + DoubleToString(lowest,  8) +
           ",\"high_idx\":" + IntegerToString(hi_idx)    +
           ",\"low_idx\":"  + IntegerToString(lo_idx)    + "}";
}

//+------------------------------------------------------------------+
//| Construieste payload JSON pentru /analyze_mtf                    |
//+------------------------------------------------------------------+
string BuildPayload(string symbol, double balance, double equity,
                    int open_trades, string kill_zone)
{
    int    digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
    double bid    = SymbolInfoDouble(symbol, SYMBOL_BID);
    double spread = SymbolInfoDouble(symbol, SYMBOL_ASK) - bid;

    string json = "{";
    json += "\"symbol\":\"" + symbol + "\",";
    json += "\"balance\":"  + DoubleToString(balance, 2)     + ",";
    json += "\"equity\":"   + DoubleToString(equity,  2)     + ",";
    json += "\"max_risk_usd\":" + DoubleToString(InpMaxRiskUSD, 2) + ",";
    json += "\"open_trades\":" + IntegerToString(open_trades) + ",";
    json += "\"kill_zone\":\"" + kill_zone + "\",";
    json += "\"current_price\":" + DoubleToString(bid, digits) + ",";
    json += "\"spread\":"   + DoubleToString(spread, digits) + ",";
    json += "\"digits\":"   + IntegerToString(digits)        + ",";

    // Bare OHLCV per timeframe (pentru analiza + grafic)
    string tf_names[] = {"M1",       "M5",       "M15",        "H1",       "H4"};
    ENUM_TIMEFRAMES tfs[] = {PERIOD_M1, PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4};

    json += "\"timeframes\":{";
    for(int i = 0; i < 5; i++)
    {
        if(i > 0) json += ",";
        json += "\"" + tf_names[i] + "\":";
        json += BarsToJSON(symbol, tfs[i], InpBarsPerTF);
    }
    json += "},";

    // Swing High/Low din 1000 candele (pentru Fibonacci precis)
    json += "\"fib_swings\":{";
    for(int i = 0; i < 5; i++)
    {
        if(i > 0) json += ",";
        json += "\"" + tf_names[i] + "\":";
        json += GetFibSwing(symbol, tfs[i], InpFibLookback);
    }
    json += "}";

    json += "}";
    return json;
}

//+------------------------------------------------------------------+
//| HTTP POST                                                         |
//+------------------------------------------------------------------+
string SendHTTPRequest(string url, string payload)
{
    char   post_data[];
    char   result_data[];
    string result_headers;

    StringToCharArray(payload, post_data, 0, StringLen(payload));
    string headers = "Content-Type: application/json\r\n";

    int res = WebRequest("POST", url, headers, 15000, post_data, result_data, result_headers);

    if(res == 200) return CharArrayToString(result_data);
    if(res == -1)  Print("[!] WebRequest eroare. Adauga ", url, " in Tools>Options>Expert Advisors>Allow WebRequest");
    else           Print("[!] HTTP ", res, " la ", url);
    return "";
}

//+------------------------------------------------------------------+
//| JSON parsare simpla                                               |
//+------------------------------------------------------------------+
string JSONGetStr(string json, string key)
{
    string s = "\"" + key + "\":\"";
    int p = StringFind(json, s);
    if(p < 0) return "";
    p += StringLen(s);
    int e = StringFind(json, "\"", p);
    if(e < 0) return "";
    return StringSubstr(json, p, e - p);
}

double JSONGetDouble(string json, string key)
{
    string s = "\"" + key + "\":";
    int p = StringFind(json, s);
    if(p < 0) return 0.0;
    p += StringLen(s);
    int e = p;
    while(e < StringLen(json))
    {
        string ch = StringSubstr(json, e, 1);
        if(ch == "," || ch == "}" || ch == "]") break;
        e++;
    }
    return StringToDouble(StringSubstr(json, p, e - p));
}

int JSONGetInt(string json, string key) { return (int)JSONGetDouble(json, key); }

//+------------------------------------------------------------------+
//| Calculeaza lot size bazat pe risc in USD                         |
//+------------------------------------------------------------------+
double CalculateLotSize(string symbol, double entry, double sl)
{
    double risk_money = InpMaxRiskUSD; // Fix $50

    double tick_val  = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
    double point     = SymbolInfoDouble(symbol, SYMBOL_POINT);

    double sl_pts = MathAbs(entry - sl) / point;
    if(sl_pts <= 0 || tick_val <= 0) return 0;

    double lots = risk_money / (sl_pts * tick_val / tick_size * point);

    double step    = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
    double min_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);

    lots = MathFloor(lots / step) * step;
    return MathMax(min_lot, MathMin(max_lot, lots));
}

//+------------------------------------------------------------------+
//| Review tehnic → trimite fiecare pozitie la /review_trade         |
//+------------------------------------------------------------------+
void ReviewOpenPositions()
{
    int total = PositionsTotal();
    if(total == 0) return;

    Print("[Review] Verific ", total, " pozitii deschise...");

    string tf_names[] = {"M1",      "M5",      "M15",       "H1",      "H4"};
    ENUM_TIMEFRAMES tfs[] = {PERIOD_M1, PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4};

    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(!g_position.SelectByIndex(i)) continue;
        if(g_position.Magic() != InpMagicNumber) continue;

        string symbol    = g_position.Symbol();
        long   ticket    = g_position.Ticket();
        int    digits    = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
        string direction = (g_position.PositionType() == POSITION_TYPE_BUY) ? "BUY" : "SELL";
        double bid       = SymbolInfoDouble(symbol, SYMBOL_BID);

        // Construieste payload pentru /review_trade
        string json = "{";
        json += "\"ticket\":"         + IntegerToString(ticket)                          + ",";
        json += "\"symbol\":\""       + symbol                                           + "\",";
        json += "\"direction\":\""    + direction                                        + "\",";
        json += "\"entry_price\":"    + DoubleToString(g_position.PriceOpen(),   digits) + ",";
        json += "\"stop_loss\":"      + DoubleToString(g_position.StopLoss(),    digits) + ",";
        json += "\"take_profit_1\":"  + DoubleToString(g_position.TakeProfit(),  digits) + ",";
        json += "\"current_price\":"  + DoubleToString(bid, digits)                      + ",";
        json += "\"digits\":"         + IntegerToString(digits)                          + ",";

        json += "\"timeframes\":{";
        for(int t = 0; t < 5; t++)
        {
            if(t > 0) json += ",";
            json += "\"" + tf_names[t] + "\":";
            json += BarsToJSON(symbol, tfs[t], InpBarsPerTF);
        }
        json += "}";
        json += "}";

        string response = SendHTTPRequest(InpServerURL + "/review_trade", json);
        if(response == "") continue;

        string action  = JSONGetStr(response, "action");
        string reason  = JSONGetStr(response, "reason");
        string cur_dir = JSONGetStr(response, "current_direction");
        int    score   = JSONGetInt(response, "confluence_score");
        int    max_s   = JSONGetInt(response, "max_score");

        Print("[Review] #", ticket, " ", symbol, " ", direction,
              " → ", action,
              " | Directie actuala: ", cur_dir,
              " | Confluenta: ", score, "/", max_s,
              " | Motiv: ", reason);

        if(action == "CLOSE")
        {
            double profit = g_position.Profit();
            if(g_trade.PositionClose(ticket))
                Print("[Review] INCHIS #", ticket, " ", symbol,
                      " | Profit: $", DoubleToString(profit, 2),
                      " | Motiv: ", reason);
            else
                Print("[Review] Eroare inchidere #", ticket, ": ", GetLastError());
        }

        Sleep(1000); // pauza intre pozitii sa nu suprasolicite serverul
    }
}

//+------------------------------------------------------------------+
//| Monitorizeaza pozitiile deschise → trimite la /monitor_mtf       |
//+------------------------------------------------------------------+
void MonitorOpenPositions()
{
    int total = PositionsTotal();
    if(total == 0) return;

    string pos_json = "[";
    bool   first    = true;

    for(int i = 0; i < total; i++)
    {
        if(!g_position.SelectByIndex(i)) continue;
        if(g_position.Magic() != InpMagicNumber) continue;

        if(!first) pos_json += ",";
        first = false;

        int    digits = (int)SymbolInfoInteger(g_position.Symbol(), SYMBOL_DIGITS);
        double risk_usd = InpMaxRiskUSD;

        pos_json += "{";
        pos_json += "\"ticket\":"  + IntegerToString(g_position.Ticket()) + ",";
        pos_json += "\"symbol\":\"" + g_position.Symbol() + "\",";
        pos_json += "\"type\":\"" + (g_position.PositionType() == POSITION_TYPE_BUY ? "BUY" : "SELL") + "\",";
        pos_json += "\"entry\":"   + DoubleToString(g_position.PriceOpen(),    digits) + ",";
        pos_json += "\"sl\":"      + DoubleToString(g_position.StopLoss(),     digits) + ",";
        pos_json += "\"tp\":"      + DoubleToString(g_position.TakeProfit(),   digits) + ",";
        pos_json += "\"profit\":"  + DoubleToString(g_position.Profit(),       2)      + ",";
        pos_json += "\"current\":" + DoubleToString(g_position.PriceCurrent(), digits) + ",";
        pos_json += "\"risk_usd\":" + DoubleToString(risk_usd, 2);
        pos_json += "}";
    }
    pos_json += "]";

    if(first) return; // fara pozitii

    string payload  = "{\"positions\":" + pos_json + "}";
    string response = SendHTTPRequest(InpServerURL + "/monitor_mtf", payload);
    if(response == "") return;

    // Executa comenzile de inchidere
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(!g_position.SelectByIndex(i)) continue;
        if(g_position.Magic() != InpMagicNumber) continue;

        string ticket_str = IntegerToString(g_position.Ticket());
        string close_key  = "\"" + ticket_str + "\":\"CLOSE\"";

        if(StringFind(response, close_key) >= 0)
        {
            double profit = g_position.Profit();
            string sym    = g_position.Symbol();
            if(g_trade.PositionClose(g_position.Ticket()))
                Print("[Monitor] Inchis ", sym, " #", ticket_str,
                      " | Profit: $", DoubleToString(profit, 2));
            else
                Print("[Monitor] Eroare inchidere #", ticket_str, ": ", GetLastError());
        }
    }
}

//+------------------------------------------------------------------+
//| Proceseaza raspunsul si plaseaza trade-ul                        |
//+------------------------------------------------------------------+
void ProcessServerResponse(string symbol, string response)
{
    string decision = JSONGetStr(response, "decision");
    int    conf     = JSONGetInt(response,    "confidence_score");
    int    confl    = JSONGetInt(response,    "confluence_score");

    Print("[", symbol, "] ", decision, " | Precizie: ", conf, "% | Confluenta: ", confl);

    if(decision == "" || decision == "HOLD") return;

    if(conf < InpMinConfidence)
    {
        Print("[", symbol, "] Precizie insuficienta (", conf, "% < ", InpMinConfidence, "%). Skip.");
        return;
    }

    double entry = JSONGetDouble(response, "entry_price");
    double sl    = JSONGetDouble(response, "stop_loss");
    double tp1   = JSONGetDouble(response, "take_profit_1");
    double tp2   = JSONGetDouble(response, "take_profit_2");
    double rr    = JSONGetDouble(response, "rr_ratio");
    string chart = JSONGetStr(response, "chart_url");

    if(entry <= 0 || sl <= 0 || tp1 <= 0) { Print("[", symbol, "] Parametri invalizi."); return; }
    if(rr < InpMinRR) { Print("[", symbol, "] R:R prea mic: ", rr); return; }

    double lots_total = CalculateLotSize(symbol, entry, sl);
    if(lots_total <= 0) { Print("[", symbol, "] Lot invalid."); return; }

    // Imparte in 2 pozitii: jumatate spre TP1, jumatate spre TP2
    double step     = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
    double min_lot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double lots_half = MathFloor(lots_total / 2.0 / step) * step;
    if(lots_half < min_lot) lots_half = min_lot;

    int    digits  = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
    string comment = "MTF_" + decision + "_R" + DoubleToString(rr, 1);
    bool   ok1 = false, ok2 = false;

    if(decision == "BUY")
    {
        ok1 = g_trade.Buy(lots_half, symbol, 0, sl, tp1, comment + "_P1");
        ok2 = (tp2 > 0) ? g_trade.Buy(lots_half, symbol, 0, sl, tp2, comment + "_P2") : false;
    }
    if(decision == "SELL")
    {
        ok1 = g_trade.Sell(lots_half, symbol, 0, sl, tp1, comment + "_P1");
        ok2 = (tp2 > 0) ? g_trade.Sell(lots_half, symbol, 0, sl, tp2, comment + "_P2") : false;
    }

    if(ok1)
    {
        Print("╔══════════════════════════════════════════════╗");
        Print("║  TRADE: ", decision, " ", symbol, " | Lots: ", lots_half, " x2");
        Print("║  Entry: ", DoubleToString(entry, digits),
              "  SL: ", DoubleToString(sl, digits));
        Print("║  TP1: ", DoubleToString(tp1, digits),
              "  TP2: ", DoubleToString(tp2, digits));
        Print("║  Risk: $", InpMaxRiskUSD, " | R:R 1:", DoubleToString(rr, 1));
        Print("║  Precizie: ", conf, "% | Grafic: ", chart);
        Print("╚══════════════════════════════════════════════╝");
    }
    else
        Print("[", symbol, "] Trade esuat: ", GetLastError());
}

//+------------------------------------------------------------------+
//| Muta SL la break-even cand profitul atinge 1R                   |
//+------------------------------------------------------------------+
void ManageBreakEven()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(!g_position.SelectByIndex(i)) continue;
        if(g_position.Magic() != InpMagicNumber) continue;

        string sym    = g_position.Symbol();
        int    digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
        double entry  = g_position.PriceOpen();
        double sl     = g_position.StopLoss();
        double tp     = g_position.TakeProfit();
        double cur    = g_position.PriceCurrent();
        double pt     = SymbolInfoDouble(sym, SYMBOL_POINT);

        if(sl <= 0) continue;

        double risk = MathAbs(entry - sl);
        if(risk <= 0) continue;

        bool is_buy = (g_position.PositionType() == POSITION_TYPE_BUY);

        // Sari daca SL e deja la sau dincolo de break-even
        if(is_buy  && sl >= entry - pt * 2) continue;
        if(!is_buy && sl <= entry + pt * 2) continue;

        // Trigger: pretul a avansat cel putin 1R in directia noastra
        bool trigger = (is_buy  && cur >= entry + risk) ||
                       (!is_buy && cur <= entry - risk);

        if(trigger)
        {
            double new_sl = NormalizeDouble(entry, digits);
            if(g_trade.PositionModify(g_position.Ticket(), new_sl, tp))
                Print("[BE] ", sym, " #", g_position.Ticket(),
                      " — SL la break-even: ", DoubleToString(new_sl, digits));
        }
    }
}

//+------------------------------------------------------------------+
void ScanAllSymbols()
{
    if(IsDailyLossLimitReached()) return;

    string kill_zone = GetKillZone();
    double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
    double equity    = AccountInfoDouble(ACCOUNT_EQUITY);

    for(int i = 0; i < g_symbol_count; i++)
    {
        // Reverifica limita la fiecare simbol (fix race condition)
        int open_trades = CountOpenTrades();
        if(open_trades >= InpMaxOpenTrades)
        {
            Print("[MultiTFTrader] Limita ", InpMaxOpenTrades, " pozitii atinsa. Stop scanare.");
            return;
        }

        string sym = g_symbols[i];
        StringTrimLeft(sym); StringTrimRight(sym);
        if(sym == "" || HasOpenPosition(sym)) continue;

        // Max 1 pozitie per valuta — evita GBPUSD + GBPJPY + GBPCAD toate odata
        string base  = StringSubstr(sym, 0, 3);
        string quote = StringSubstr(sym, 3, 3);
        if(CountCurrencyExposure(base) >= 1 || CountCurrencyExposure(quote) >= 1)
        {
            // Permite max 1 expunere per valuta (ex: 1 trade cu USD, 1 cu EUR)
            // Sari daca acea valuta e deja implicata intr-un alt trade
            continue;
        }

        Print("[MultiTFTrader] Analizez: ", sym, " (pozitii deschise: ", open_trades, "/", InpMaxOpenTrades, ")");
        string payload  = BuildPayload(sym, balance, equity, open_trades, kill_zone);
        string response = SendHTTPRequest(InpServerURL + "/analyze_mtf", payload);

        if(response != "")
        {
            int before = CountOpenTrades();
            ProcessServerResponse(sym, response);
            int after = CountOpenTrades();

            // Daca s-a deschis un trade nou, asteapta confirmare si re-verifica
            if(after > before)
            {
                Sleep(3000);
                if(CountOpenTrades() >= InpMaxOpenTrades)
                {
                    Print("[MultiTFTrader] Limita atinsa dupa trade. Opresc scanarea.");
                    return;
                }
            }
        }

        Sleep(2000);
    }
}
