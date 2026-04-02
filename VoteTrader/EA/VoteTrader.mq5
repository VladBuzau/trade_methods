//+------------------------------------------------------------------+
//|  VoteTrader — MQL5 Service                                       |
//|  12 metode de analiza, vot majoritar                             |
//+------------------------------------------------------------------+
#property service
#property copyright "VoteTrader"
#property version   "1.00"

#include <Trade\Trade.mqh>

input string InpServerURL  = "https://postlarval-barb-delineative.ngrok-free.dev/signal";
input string InpSymbols    = "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,EURGBP,EURJPY,GBPJPY,AUDJPY,XAUUSD";
input double InpRiskDollar = 50.0;
input int    InpMaxTrades  = 5;
input int    InpCheckSec   = 60;
input int    InpMagic      = 202700;

CTrade g_trade;

void OnStart()
{
    g_trade.SetExpertMagicNumber(InpMagic);
    g_trade.SetDeviationInPoints(20);
    Print("VoteTrader Service pornit. Magic=", InpMagic);

    while(!IsStopped())
    {
        ManageBreakEven();
        if(CountPositions() < InpMaxTrades)
            ScanSymbols();
        Sleep(InpCheckSec * 1000);
    }
    Print("VoteTrader Service oprit.");
}

int CountPositions()
{
    int cnt = 0;
    for(int i = PositionsTotal()-1; i >= 0; i--)
    {
        ulong t = PositionGetTicket(i);
        if(PositionSelectByTicket(t) && (long)PositionGetInteger(POSITION_MAGIC) == (long)InpMagic)
            cnt++;
    }
    return cnt;
}

bool HasPosition(string sym)
{
    for(int i = PositionsTotal()-1; i >= 0; i--)
    {
        ulong t = PositionGetTicket(i);
        if(PositionSelectByTicket(t) &&
           (long)PositionGetInteger(POSITION_MAGIC) == (long)InpMagic &&
           PositionGetString(POSITION_SYMBOL) == sym)
            return true;
    }
    return false;
}

void ScanSymbols()
{
    string syms[];
    int n = StringSplit(InpSymbols, ',', syms);
    for(int i = 0; i < n; i++)
    {
        if(IsStopped() || CountPositions() >= InpMaxTrades) break;
        string sym = syms[i];
        StringTrimRight(sym); StringTrimLeft(sym);
        if(sym == "" || HasPosition(sym)) continue;
        RequestSignal(sym);
        Sleep(600);
    }
}

void RequestSignal(string symbol)
{
    string url     = InpServerURL + "?symbol=" + symbol;
    string headers = "Content-Type: application/json\r\nngrok-skip-browser-warning: true\r\n";
    char   post[], result[];
    string res_headers;
    int code = WebRequest("GET", url, headers, 8000, post, result, res_headers);
    if(code != 200) { Print("WebRequest error ", code, " for ", symbol); return; }
    ProcessResponse(symbol, CharArrayToString(result));
}

string JStr(string j, string k)
{
    string s = "\""+k+"\":\"";
    int p = StringFind(j,s); if(p<0) return "";
    p += StringLen(s);
    int e = StringFind(j,"\"",p); if(e<0) return "";
    return StringSubstr(j,p,e-p);
}

double JNum(string j, string k)
{
    string s = "\""+k+"\":";
    int p = StringFind(j,s); if(p<0) return 0;
    p += StringLen(s);
    while(p<StringLen(j) && StringGetCharacter(j,p)==' ') p++;
    string n="";
    while(p<StringLen(j)){
        ushort c=StringGetCharacter(j,p);
        if(c==','||c=='}'||c=='\n'||c=='\r') break;
        n+=StringSubstr(j,p,1); p++;
    }
    return StringToDouble(n);
}

void ProcessResponse(string symbol, string json)
{
    string dir = JStr(json, "direction");
    double conf= JNum(json, "confidence");
    double sl  = JNum(json, "sl");
    double tp  = JNum(json, "tp");
    double price=JNum(json, "price");
    int    bvotes=(int)JNum(json,"votes_buy");
    int    svotes=(int)JNum(json,"votes_sell");

    if(dir=="HOLD"||dir=="")
    {
        Print(symbol," HOLD (",DoubleToString(conf,1),"%) BUY=",bvotes," SELL=",svotes);
        return;
    }
    if(sl==0.0||tp==0.0){ Print(symbol," SL/TP invalid, skip"); return; }

    Print(symbol," SEMNAL: ",dir," conf=",DoubleToString(conf,1),"% votes=",
          (dir=="BUY"?bvotes:svotes),"/12 sl=",DoubleToString(sl,5)," tp=",DoubleToString(tp,5));

    double tick_val  = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
    double sl_dist   = MathAbs(price - sl);
    if(sl_dist<=0||tick_size<=0||tick_val<=0) return;

    double lot_step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
    double min_lot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double max_lot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
    double lots     = MathFloor(InpRiskDollar/(sl_dist/tick_size*tick_val)/lot_step)*lot_step;
    if(lots < min_lot) lots = min_lot;
    if(lots > max_lot) lots = max_lot;

    bool ok;
    string comment = "VT_" + dir + "_" + IntegerToString(bvotes+svotes);
    if(dir=="BUY")  ok = g_trade.Buy(lots,  symbol, 0, sl, tp, comment);
    else            ok = g_trade.Sell(lots, symbol, 0, sl, tp, comment);

    Print(symbol," pozitie: ",ok," lots=",DoubleToString(lots,2));
}

void ManageBreakEven()
{
    for(int i=PositionsTotal()-1;i>=0;i--)
    {
        ulong t=PositionGetTicket(i);
        if(!PositionSelectByTicket(t)) continue;
        if((long)PositionGetInteger(POSITION_MAGIC)!=(long)InpMagic) continue;
        double entry=PositionGetDouble(POSITION_PRICE_OPEN);
        double sl   =PositionGetDouble(POSITION_SL);
        double tp   =PositionGetDouble(POSITION_TP);
        double cur  =PositionGetDouble(POSITION_PRICE_CURRENT);
        bool is_buy =(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY);
        double risk =MathAbs(entry-sl);
        if(risk<=0) continue;
        if(is_buy&&sl>=entry) continue;
        if(!is_buy&&sl<=entry) continue;
        bool trigger=(is_buy&&cur>=entry+risk)||(!is_buy&&cur<=entry-risk);
        if(trigger){ g_trade.PositionModify(t,entry,tp); Print("BE: ",t); }
    }
}
//+------------------------------------------------------------------+
