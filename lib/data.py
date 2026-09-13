import yfinance as yf
import pandas as pd
from datetime import datetime

SYMBOL_MAP = {
    "XAUUSD": "GC=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "BOOM1000": None,      # Deriv needed
    "CRASH1000": None,     # Deriv needed
}

def get_candles(symbol: str, period="5d", interval="1h", last_n=12):
    """Return last N hourly OHLCV candles + current price"""
    yf_symbol = SYMBOL_MAP.get(symbol)
    
    if yf_symbol is None:
        # Boom / Crash placeholder
        return {
            "price": None,
            "candles_text": "No free public OHLCV available. Add Deriv App ID for full analysis.",
            "ok": False
        }
    
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return {"price": None, "candles_text": "No data", "ok": False}
        
        df = df.tail(last_n)
        price = round(float(df["Close"].iloc[-1]), 5 if "USD" in symbol else 2)
        
        lines = []
        for idx, row in df.iterrows():
            ts = idx.strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"{ts} | O:{row['Open']:.5f} H:{row['High']:.5f} "
                f"L:{row['Low']:.5f} C:{row['Close']:.5f} V:{int(row['Volume'])}"
            )
        
        return {
            "price": price,
            "candles_text": "\n".join(lines),
            "ok": True
        }
    except Exception as e:
        return {"price": None, "candles_text": str(e), "ok": False} 