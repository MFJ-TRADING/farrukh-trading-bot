from http.server import BaseHTTPRequestHandler
import os
import requests
from lib.data import get_candles
from lib.analysis import analyze_with_taurus

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK")
SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "BOOM1000", "CRASH1000"]

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        results = []
        
        for symbol in SYMBOLS:
            data = get_candles(symbol)
            
            if not data["ok"] or data["price"] is None:
                # Limited message for Boom/Crash
                msg = f"""📊 **TAURUS AI — {symbol}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Status:** Limited data (Deriv App ID required for full OHLCV)
**Action:** WAIT / Monitor only
⚠️ Add DERIV_APP_ID to enable full analysis.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
            else:
                msg = analyze_with_taurus(
                    symbol=symbol,
                    price=data["price"],
                    candles_data=data["candles_text"]
                )
            
            # Send to Discord
            if DISCORD_WEBHOOK:
                try:
                    requests.post(DISCORD_WEBHOOK, json={"content": msg}, timeout=10)
                    results.append(f"{symbol}: OK")
                except Exception as e:
                    results.append(f"{symbol}: Discord error - {e}")
            else:
                results.append(f"{symbol}: No webhook")
        
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write("\n".join(results).encode())
        return 