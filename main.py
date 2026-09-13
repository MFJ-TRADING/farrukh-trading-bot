import os
import time
import requests
import yfinance as yf
from groq import Groq
from datetime import datetime, timedelta

# ====================== CONFIG ======================
SYMBOLS = {
    "XAUUSD": "GC=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "mf-trading-bot-90")
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"

client = Groq(api_key=GROQ_API_KEY)


# ====================== NOTIFICATION ======================
def send_to_ntfy(message, title="Trading Signal"):
    """Send a plain-text notification to ntfy.sh topic."""
    try:
        response = requests.post(
            NTFY_URL,
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": "default",
            },
            timeout=15,
        )
        print(f"ntfy Status: {response.status_code}")
        if response.status_code != 200:
            print("ntfy Error:", response.text)
        return response.status_code == 200
    except Exception as e:
        print("ntfy exception:", e)
        return False


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

📊 {symbol} — Next 1H Candle Prediction
━━━━━━━━━━━━━━━━━━━━
Current Price: {price}

Next Candle: Bullish / Bearish
Expected High: [price]
Expected Low: [price]

Reason: 1 short sentence only.
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ {symbol} — AI analysis failed: {e}"


# ====================== CORE JOB ======================
def run_analysis_cycle():
    print(f"\n=== Running analysis cycle at {datetime.utcnow()} UTC ===")

    for name, yf_sym in SYMBOLS.items():
        print(f"\nProcessing {name}...")
        price, candles = get_candles(yf_sym)

        if price is None:
            msg = f"⚠️ {name} — Data not available ({candles})"
        else:
            msg = analyze(name, price, candles)

        send_to_ntfy(msg, title=f"{name} Signal")
        time.sleep(1)  # small gap so notifications don't overlap/collapse

    print("\nCycle complete.")


# ====================== SCHEDULER ======================
def seconds_until_next_hour():
    now = datetime.now()
    next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    return (next_hour - now).total_seconds()


def main():
    print(f"Bot starting at {datetime.utcnow()} UTC")

    # 1. Send an immediate test notification so you know it's alive
    test_ok = send_to_ntfy(
        "✅ Bot started successfully. Waiting for the next hour to begin analysis...",
        title="Bot Test Notification",
    )
    if not test_ok:
        print("WARNING: Test notification failed to send. Check NTFY_TOPIC / internet connection.")

    # 2. Wait until the top of the next hour, then run every hour after that
    while True:
        wait_secs = seconds_until_next_hour()
        mins = int(wait_secs // 60)
        secs = int(wait_secs % 60)
        print(f"Waiting {mins}m {secs}s until next hour starts...")
        time.sleep(wait_secs)

        try:
            run_analysis_cycle()
        except Exception as e:
            error_msg = f"⚠️ Bot crashed during cycle: {e}"
            print(error_msg)
            send_to_ntfy(error_msg, title="Bot Error")

        # tiny buffer so we don't fire twice at the boundary due to timing drift
        time.sleep(2)


if __name__ == "__main__":
    main()
