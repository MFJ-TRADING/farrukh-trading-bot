import os
import requests
import yfinance as yf
from groq import Groq
from datetime import datetime

# ====================== CONFIG ======================
SYMBOLS = {
    "XAUUSD": "GC=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "SOLUSD": "SOL-USD",
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK")

client = Groq(api_key=GROQ_API_KEY)

# ====================== DATA ======================
def get_candles(yf_symbol, last_n=12):
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="5d", interval="1h")
        
        if df.empty:
            return None, "No data available"
        
        df = df.tail(last_n)
        price = round(float(df["Close"].iloc[-1]), 5 if "X" in yf_symbol else 2)
        
        lines = []
        for idx, row in df.iterrows():
            ts = idx.strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"{ts} | O:{row['Open']:.5f} H:{row['High']:.5f} "
                f"L:{row['Low']:.5f} C:{row['Close']:.5f}"
            )
        return price, "\n".join(lines)
    except Exception as e:
        return None, str(e)

# ====================== AI ANALYSIS ======================
def analyze(symbol, price, candles):
    prompt = f"""
You are Taurus AI — an elite Senior Quantitative Financial Analyst specializing in Gold and major FX pairs.

INPUT:
- Asset: {symbol}
- Timeframe: 1H
- Current Price: {price}
- Recent Hourly Candles:
{candles}

TASK:
1. Analyze price action, market structure, liquidity.
2. Give clear decision: [BUY], [SELL] or [WAIT]
3. Give Entry, SL, TP1, TP2 and Risk:Reward
4. Next hour candle prediction buy aur sell.

OUTPUT strictly in this Discord Markdown format:

📊 **TAURUS AI — INSTITUTIONAL MARKET ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Asset:** {symbol} | **TF:** 1H | **Price:** {price}
**Trigger:** Hourly Bar Close

🎯 **TRADE DIRECTIVE:** `[BUY / SELL / WAIT]`
⚖️ **CONFIDENCE LEVEL:** `[High / Medium / Low]`
 next hour candle prediction
📍 **KEY TRADE LEVELS:**
• **Entry Zone:** `[zone]`
• **Stop Loss (SL):** `[SL]`
• **Take Profit 1 (TP1):** `[TP1]`
• **Take Profit 2 (TP2):** `[TP2]`
• **Risk-to-Reward:** `[e.g. 1:2]`

🔍 **TECHNICAL RATIONALE:**
• **Market Structure:** [1-2 sentences]
• **Price Action & Liquidity:** [1-2 sentences]
• **Risk Assessment:** [1 sentence]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ *Risk Warning: Always manage position size. Not financial advice.*
"""

    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=900
    )
    return response.choices[0].message.content

# ====================== SEND TO DISCORD ======================
def send_to_discord(message):
    if not DISCORD_WEBHOOK:
        print("No Discord webhook set")
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": message}, timeout=15)
        print("Message sent to Discord")
    except Exception as e:
        print("Discord error:", e)

# ====================== MAIN ======================
def main():
    print(f"Starting analysis at {datetime.utcnow()} UTC")
    
    for name, yf_sym in SYMBOLS.items():
        print(f"\nProcessing {name}...")
        price, candles = get_candles(yf_sym)
        
        if price is None:
            msg = f"⚠️ **{name}** — Data not available right now.\n{candles}"
        else:
            msg = analyze(name, price, candles)
        
        send_to_discord(msg)
    
    print("\nAll done.")

if __name__ == "__main__":
    main()
