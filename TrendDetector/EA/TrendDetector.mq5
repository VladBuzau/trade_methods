//+------------------------------------------------------------------+
//|  TrendDetector — MQL5 Service                                    |
//|  Ruleaza in background, fara chart, nu se opreste niciodata      |
//+------------------------------------------------------------------+
#property service
#property copyright "TrendDetector"
#property version   "1.00"

#include <Trade\Trade.mqh>

//── Inputs ──────────────────────────────────────────────────────────
input string InpServerURL   = "https://postlarval-barb-delineative.ngrok-free.dev/signal";
input string InpSymbols     = "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,EURGBP,EURJPY,GBPJPY,AUDJPY,XAUUSD";
input double InpRiskDollar  = 50.0;   // Risc fix per trade ($)
input int    InpMaxTrades   = 5;      // Max pozitii simultan
input int    InpCheckSec    = 60;
input int    InpMagic       = 202600;

//── Globals ─────────────────────────────────────────────────────────
CTrade g_trade;

//+------------------------------------------------------------------+
void OnStart()
{
    g_trade.SetExpertMagicNumber(InpMagic);
    g_trade.SetDeviationInPoints(20);
    Print("TrendDetector Service pornit. Magic=", InpMagic);

    while(!IsStopped())
    {
        ManageBreakEven();

        if(CountMagicPositions() < InpMaxTrades)
            CheckAllSymbols();

        Sleep(InpCheckSec * 1000);
    }

    Print("TrendDetector Service oprit.");
}

//+------------------------------------------------------------------+
int CountMagicPositions()
{
    int cnt = 0;
    for(int i = PositionsTotal()-1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket) &&
           (long)PositionGetInteger(POSITION_MAGIC) == (long)InpMagic)
            cnt++;
    }
    return cnt;
}

bool SymbolHasOpenPosition(string symbol)
{
    for(int i = PositionsTotal()-1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket) &&
           (long)PositionGetInteger(POSITION_MAGIC) == (long)InpMagic &&
           PositionGetString(POSITION_SYMBOL) == symbol)
            return true;
    }
    return false;
}

//+------------------------------------------------------------------+
void CheckAllSymbols()
{
    string symbols[];
    int n = StringSplit(InpSymbols, ',', symbols);
    for(int i = 0; i < n; i++)
    {
        if(IsStopped()) break;
        if(CountMagicPositions() >= InpMaxTrades) break;

        string sym = symbols[i];
        StringTrimRight(sym);
        StringTrimLeft(sym);
        if(sym == "") continue;
        if(SymbolHasOpenPosition(sym)) continue;

        RequestSignal(sym);
        Sleep(500); // pauza intre requesturi
    }
}

//+------------------------------------------------------------------+
void RequestSignal(string symbol)
{
    string url     = InpServerURL + "?symbol=" + symbol;
    string headers = "Content-Type: application/json\r\nngrok-skip-browser-warning: true\r\n";
    char   post[];
    char   result[];
    string res_headers;

    int code = WebRequest("GET", url, headers, 8000, post, result, res_headers);
    if(code != 200)
    {
        Print("WebRequest error ", code, " for ", symbol);
        return;
    }

    string json = CharArrayToString(result);
    ProcessResponse(symbol, json);
}

//+------------------------------------------------------------------+
string JsonStr(string json, string key)
{
    string search = "\"" + key + "\":\"";
    int pos = StringFind(json, search);
    if(pos < 0) return "";
    pos += StringLen(search);
    int end = StringFind(json, "\"", pos);
    if(end < 0) return "";
    return StringSubstr(json, pos, end - pos);
}

double JsonNum(string json, string key)
{
    string search = "\"" + key + "\":";
    int pos = StringFind(json, search);
    if(pos < 0) return 0.0;
    pos += StringLen(search);
    while(pos < StringLen(json) && StringGetCharacter(json, pos) == ' ') pos++;
    string num = "";
    while(pos < StringLen(json))
    {
        ushort c = StringGetCharacter(json, pos);
        if(c == ',' || c == '}' || c == '\n' || c == '\r') break;
        num += StringSubstr(json, pos, 1);
        pos++;
    }
    return StringToDouble(num);
}

//+------------------------------------------------------------------+
void ProcessResponse(string symbol, string json)
{
    string direction  = JsonStr(json, "direction");
    double confidence = JsonNum(json, "confidence");
    double sl         = JsonNum(json, "sl");
    double tp1        = JsonNum(json, "tp1");
    double tp2        = JsonNum(json, "tp2");
    double price      = JsonNum(json, "price");

    if(direction == "HOLD" || direction == "")
    {
        Print(symbol, " HOLD (", DoubleToString(confidence, 1), "%)");
        return;
    }
    if(sl == 0.0 || tp1 == 0.0)
    {
        Print(symbol, " SL/TP invalid, skip");
        return;
    }

    Print(symbol, " SEMNAL: ", direction,
          " conf=", DoubleToString(confidence,1), "%",
          " sl=", DoubleToString(sl,5),
          " tp1=", DoubleToString(tp1,5),
          " tp2=", DoubleToString(tp2,5));

    // Calcul lot bazat pe $50 fix per trade
    double tick_val   = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size  = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
    double sl_dist    = MathAbs(price - sl);
    if(sl_dist <= 0 || tick_size <= 0 || tick_val <= 0) return;

    double lot_step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
    double min_lot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double max_lot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
    double lots     = MathFloor(InpRiskDollar / (sl_dist / tick_size * tick_val) / lot_step) * lot_step;
    if(lots < min_lot) lots = min_lot;
    if(lots > max_lot) lots = max_lot;

    string comment = "TD_" + direction;
    bool ok;

    if(direction == "BUY")
        ok = g_trade.Buy(lots, symbol, 0, sl, tp1, comment);
    else
        ok = g_trade.Sell(lots, symbol, 0, sl, tp1, comment);

    Print(symbol, " pozitie: ", ok, " lots=", DoubleToString(lots,2), " sl=", DoubleToString(sl,5), " tp=", DoubleToString(tp1,5));
}

//+------------------------------------------------------------------+
void ManageBreakEven()
{
    for(int i = PositionsTotal()-1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(!PositionSelectByTicket(ticket)) continue;
        if((long)PositionGetInteger(POSITION_MAGIC) != (long)InpMagic) continue;

        double entry  = PositionGetDouble(POSITION_PRICE_OPEN);
        double sl     = PositionGetDouble(POSITION_SL);
        double tp     = PositionGetDouble(POSITION_TP);
        double cur    = PositionGetDouble(POSITION_PRICE_CURRENT);
        bool   is_buy = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
        double risk   = MathAbs(entry - sl);

        if(risk <= 0) continue;
        if(is_buy  && sl >= entry) continue;
        if(!is_buy && sl <= entry) continue;

        bool trigger = (is_buy  && cur >= entry + risk) ||
                       (!is_buy && cur <= entry - risk);

        if(trigger)
        {
            g_trade.PositionModify(ticket, entry, tp);
            Print("Break-even: ticket=", ticket, " entry=", DoubleToString(entry,5));
        }
    }
}
//+------------------------------------------------------------------+
