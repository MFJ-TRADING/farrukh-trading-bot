import os
import time
import requests
import yfinance as yf
from groq import Groq
from datetime import datetime, timedelta

# ============================================================
# CONFIG
# ============================================================
SYMBOLS = {
    "XAUUSD": {"yf": "GC=F", "type": "futures"},   # Gold futures - closed weekends
    "EURUSD": {"yf": "EURUSD=X", "type": "forex"}, # Forex - closed weekends
    "GBPUSD": {"yf": "GBPUSD=X", "type": "forex"}, # Forex - closed weekends
    "BTCUSD": {"yf": "BTC-USD", "type": "crypto"}, # Crypto - 24/7
    "ETHUSD": {"yf": "ETH-USD", "type": "crypto"}, # Crypto - 24/7
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "mf-trading-bot-90")
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"

GROQ_MODEL = "openai/gpt-oss-20b"
MAX_AI_RETRIES = 2          # retry AI call this many times if empty response
MAX_NTFY_RETRIES = 2        # retry sending notification if it fails
SLEEP_BETWEEN_SYMBOLS = 2   # seconds, avoid rate limits / overlapping notifications

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# LOGGING
# ============================================================
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


# ============================================================
# MARKET STATUS
# ============================================================
def is_forex_market_closed():
    """
    Forex/Gold futures market is closed roughly:
    Friday ~21:00 UTC to Sunday ~21:00 UTC.
    This is an approximation good enough to avoid false 'data error' alerts.
    """
    now_utc = datetime.utcnow()
    weekday = now_utc.weekday()  # Monday=0 ... Sunday=6
    hour = now_utc.hour

    if weekday == 5:  # Saturday - fully closed
        return True
    if weekday == 6 and hour < 21:  # Sunday before ~21:00 UTC - still closed
        return True
    if weekday == 4 and hour >= 21:  # Friday after ~21:00 UTC - closed
        return True
    return False


# ============================================================
# NOTIFICATIONS (ntfy.sh) with retry + empty-message protection
# ============================================================
def send_to_ntfy(message, title="Trading Signal"):
    if not message or not message.strip():
        message = f"⚠️ {title}: (empty message - check logs)"
        log(f"WARNING: empty message replaced with placeholder for '{title}'")

    for attempt in range(1, MAX_NTFY_RETRIES + 1):
        try:
            response = requests.post(
                NTFY_URL,
                data=message.encode("utf-8"),
                headers={"Title": title, "Priority": "default"},
                timeout=15,
            )
            if response.status_code == 200:
                log(f"ntfy OK ({title})")
                return True
            else:
                log(f"ntfy attempt {attempt} failed: {response.status_code} {response.text}")
        except Exception as e:
            log(f"ntfy attempt {attempt} exception: {e}")

        if attempt < MAX_NTFY_RETRIES:
            time.sleep(3)

    log(f"ntfy FAILED after {MAX_NTFY_RETRIES} attempts for '{title}'")
    return False


# ============================================================
# DATA FETCH
# ============================================================
def get_candles(yf_symbol, last_n=10):
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="5d", interval="1h")

        if df.empty:
            return None, "No data returned from yfinance"

        df = df.tail(last_n)
        if df.empty:
            return None, "Not enough candle data"

        price = round(float(df["Close"].iloc[-1]), 5 if "=X" in yf_symbol else 2)

        lines = []
        for idx, row in df.iterrows():
            ts = idx.strftime("%H:%M")
            lines.append(
                f"{ts} | O:{row['Open']:.5f} H:{row['High']:.5f} "
                f"L:{row['Low']:.5f} C:{row['Close']:.5f}"
            )
        return price, "\n".join(lines)
    except Exception as e:
        return None, f"Exception: {e}"


# ============================================================
# AI ANALYSIS with retry on empty response
# ============================================================
def analyze(symbol, price, candles):
    prompt = f"""You are a professional price action trader.

Asset: {symbol}
Current Price: {price}
Last 10 Hourly Candles:
{candles}

Predict the NEXT 1H candle only.

Reply STRICTLY in this format (nothing else, no markdown code fences):

📊 {symbol} — Next 1H Candle Prediction
Current Price: {price}
Next Candle: Bullish / Bearish
Expected High: [price]
Expected Low: [price]
Reason: 1 short sentence only.
"""

    last_error = None

    for attempt in range(1, MAX_AI_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300,
            )
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            if content and content.strip():
                # strip accidental markdown code fences
                cleaned = content.strip().strip("`").strip()
                return cleaned

            last_error = f"Empty content (finish_reason={finish_reason})"
            log(f"{symbol} AI attempt {attempt}: {last_error}")

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            log(f"{symbol} AI attempt {attempt} exception: {last_error}")

        if attempt < MAX_AI_RETRIES:
            time.sleep(3)

    return (
        f"⚠️ {symbol} — AI analysis failed after {MAX_AI_RETRIES} attempts.\n"
        f"Price: {price}\n"
        f"Last error: {last_error}"
    )


# ============================================================
# CORE ANALYSIS CYCLE
# ============================================================
def run_analysis_cycle():
    log("=" * 50)
    log("Starting analysis cycle")
    log("=" * 50)

    forex_closed = is_forex_market_closed()

    for name, info in SYMBOLS.items():
        yf_sym = info["yf"]
        asset_type = info["type"]

        log(f"Processing {name} ({asset_type})...")

        # Skip forex/futures gracefully during weekend closure
        if asset_type in ("forex", "futures") and forex_closed:
            msg = (
                f"💤 {name} — Market Closed\n"
                f"Forex/Futures market is closed (weekend). "
                f"Next analysis will resume when market reopens."
            )
            send_to_ntfy(msg, title=f"{name} - Market Closed")
            time.sleep(SLEEP_BETWEEN_SYMBOLS)
            continue

        price, candles = get_candles(yf_sym)

        if price is None:
            msg = f"⚠️ {name} — Data not available.\nReason: {candles}"
            send_to_ntfy(msg, title=f"{name} - Data Error")
        else:
            msg = analyze(name, price, candles)
            send_to_ntfy(msg, title=f"{name} Signal")

        time.sleep(SLEEP_BETWEEN_SYMBOLS)

    log("Cycle complete.")


# ============================================================
# SCHEDULER
# ============================================================
def seconds_until_next_hour():
    now = datetime.now()
    next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    return (next_hour - now).total_seconds()


def main():
    log("Bot starting...")

    if not GROQ_API_KEY:
        log("FATAL: GROQ_API_KEY is not set. Set it as an environment variable.")
        send_to_ntfy(
            "❌ Bot failed to start: GROQ_API_KEY is missing.",
            title="Bot Startup Error",
        )
        return

    # 1. Startup test notification
    send_to_ntfy(
        "✅ Bot started successfully. Running last hour's analysis now...",
        title="Bot Test Notification",
    )

    # 2. Immediate analysis using the most recent completed hourly candle
    try:
        run_analysis_cycle()
    except Exception as e:
        error_msg = f"⚠️ Bot crashed during startup cycle: {e}"
        log(error_msg)
        send_to_ntfy(error_msg, title="Bot Error")

    # 3. Hourly loop forever — never let one bad cycle kill the bot
    while True:
        wait_secs = seconds_until_next_hour()
        mins = int(wait_secs // 60)
        secs = int(wait_secs % 60)
        log(f"Waiting {mins}m {secs}s until next hour...")

        try:
            time.sleep(wait_secs)
        except Exception as e:
            log(f"Sleep interrupted: {e}")

        try:
            run_analysis_cycle()
        except Exception as e:
            error_msg = f"⚠️ Bot crashed during hourly cycle: {e}"
            log(error_msg)
            try:
                send_to_ntfy(error_msg, title="Bot Error")
            except Exception:
                pass

        time.sleep(2)  # buffer to avoid double-firing at boundary


if __name__ == "__main__":
    while True:
        try:
            main()
            # main() only returns on fatal config error (missing key) - stop retrying that
            break
        except Exception as e:
            log(f"FATAL uncaught exception, restarting in 30s: {e}")
            try:
                send_to_ntfy(f"❌ Bot crashed unexpectedly: {e}\nRestarting in 30s...", title="Bot Crash")
            except Exception:
                pass
            time.sleep(30)
            
