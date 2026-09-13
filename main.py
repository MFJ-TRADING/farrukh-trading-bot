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
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK")

client = Groq(api_key=GROQ_API_KEY)

# ====================== DATA ======================
def get_candles(yf_symbol, last_n=10):
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="5d", interval="1h")
        
        if df.empty:
            return None, "No data"
        
        df = df.tail(last_n)
        price = round(float(df["Close"].iloc[-1]), 5 if "X" in yf_symbol else 2)
        
        lines = []
        for idx, row in df.iterrows():
            ts = idx.strftime("%H:%M")
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
You are a professional price action trader.

Asset: {symbol}
Current Price: {price}
Last 10 Hourly Candles:
{candles}

Predict the NEXT 1H candle only.

Reply STRICTLY in this format (nothing else):

📊 **{symbol} — Next 1H Candle Prediction**
━━━━━━━━━━━━━━━━━━━━
**Current Price:** {price}

🟢/🔴 **Next Candle:** Bullish / Bearish
📈 **Expected High:** [price]
📉 **Expected Low:** [price]

**Reason:** 1 short sentence only.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content

# ====================== DISCORD ======================
def send_to_discord(message):
    if not DISCORD_WEBHOOK:
        print("No Discord webhook set")
        return
    
    if len(message) > 1900:
        message = message[:1900] + "\n...(truncated)"
    
    try:
        response = requests.post(
            DISCORD_WEBHOOK,
            json={"content": message},
            timeout=15
        )
        print(f"Discord Status: {response.status_code}")
        if response.status_code == 204:
            print("Message sent successfully")
        else:
            print("Discord Error:", response.text)
    except Exception as e:
        print("Discord exception:", e)

# ====================== MAIN ======================
def main():
    print(f"Starting at {datetime.utcnow()} UTC")
    
    for name, yf_sym in SYMBOLS.items():
        print(f"\nProcessing {name}...")
        price, candles = get_candles(yf_sym)
        
        if price is None:
            msg = f"⚠️ **{name}** — Data not available"
        else:
            msg = analyze(name, price, candles)
        
        send_to_discord(msg)
    
    print("\nAll done.")

if __name__ == "__main__":
    main()
