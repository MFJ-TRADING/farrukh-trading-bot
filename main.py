import os
import time
import json
import re
import requests
import yfinance as yf
from groq import Groq
from datetime import datetime, timedelta

# ============================================================
# CONFIG
# ============================================================
SYMBOLS = {
    "XAUUSD": {"yf": "GC=F", "type": "futures"},
}

# User requested Google_API_KEY name
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GROQ_API_KEY")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "mf-trading-bot-90")
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"

MODEL_PREFERENCE = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "qwen/qwen3.8-27b",
    "llama-3.1-8b-instant",
]

MAX_NTFY_RETRIES = 2
SLEEP_BETWEEN_SYMBOLS = 2
AI_MAX_TOKENS = 400
REASONING_MODEL_MAX_TOKENS = 1200

client = Groq(api_key=API_KEY)

CANDIDATE_MODELS = []
CURRENT_MODEL_IDX = 0
JSON_MODE_UNSUPPORTED = set()


def is_reasoning_model(model_name):
    low = model_name.lower()
    return "gpt-oss" in low or "deepseek" in low or "r1" in low


def extract_json_object(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])

    raise json.JSONDecodeError("No valid JSON object found", text, 0)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def discover_models():
    global CANDIDATE_MODELS
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=15,
        )
        if resp.status_code != 200:
            CANDIDATE_MODELS = list(MODEL_PREFERENCE)
            return

        available_ids = [m["id"] for m in resp.json().get("data", [])]
        ordered = [m for m in MODEL_PREFERENCE if m in available_ids]

        for mid in available_ids:
            low = mid.lower()
            if mid not in ordered and not any(x in low for x in ("whisper", "guard", "tts", "compound")):
                ordered.append(mid)

        CANDIDATE_MODELS = ordered or list(MODEL_PREFERENCE)
        log(f"Model candidates: {CANDIDATE_MODELS}")
    except Exception as e:
        log(f"Model discovery failed: {e}")
        CANDIDATE_MODELS = list(MODEL_PREFERENCE)


def get_active_model():
    if not CANDIDATE_MODELS:
        return MODEL_PREFERENCE[0]
    return CANDIDATE_MODELS[CURRENT_MODEL_IDX % len(CANDIDATE_MODELS)]


def advance_model():
    global CURRENT_MODEL_IDX
    CURRENT_MODEL_IDX += 1
    log(f"Switching to: {get_active_model()}")


def is_forex_market_closed():
    now_utc = datetime.utcnow()
    weekday = now_utc.weekday()
    hour = now_utc.hour
    if weekday == 5:
        return True
    if weekday == 6 and hour < 21:
        return True
    if weekday == 4 and hour >= 21:
        return True
    return False


def send_to_ntfy(message, title="XAUUSD Signal"):
    if not message or not message.strip():
        message = f"⚠️ {title}: (empty message)"
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
            log(f"ntfy attempt {attempt} failed: {response.status_code}")
        except Exception as e:
            log(f"ntfy attempt {attempt} exception: {e}")
        if attempt < MAX_NTFY_RETRIES:
            time.sleep(3)
    return False


def get_candles(yf_symbol, last_n=12):
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="5d", interval="1h")
        if df.empty:
            return None, "No data returned"
        df = df.tail(last_n)
        price = round(float(df["Close"].iloc[-1]), 2)

        lines = []
        for idx, row in df.iterrows():
            ts = idx.strftime("%m-%d %H:%M")
            lines.append(
                f"{ts} | O:{row['Open']:.2f} H:{row['High']:.2f} "
                f"L:{row['Low']:.2f} C:{row['Close']:.2f}"
            )
        return price, "\n".join(lines)
    except Exception as e:
        return None, str(e)


def build_prediction_message(symbol, price, data):
    direction = str(data.get("direction", "unknown")).strip().lower()
    emoji = "🟢" if direction == "bullish" else ("🔴" if direction == "bearish" else "⚪")

    lines = [
        f"📊 **TAURUS AI — XAUUSD Next 1H Prediction**",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"**Current Price:** {price}",
        f"{emoji} **Next Candle:** {direction.upper()}",
        f"📈 **Expected High:** {data.get('expected_high', 'N/A')}",
        f"📉 **Expected Low:** {data.get('expected_low', 'N/A')}",
        f"🎯 **Confidence:** {data.get('confidence', 'N/A')}%",
        f"**Reason:** {data.get('reason', 'No reason provided.')}",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "⚠️ *Not financial advice. Manage risk.*"
    ]
    return "\n".join(lines)


def analyze(symbol, price, candles):
    system_prompt = (
        "You are Taurus AI — an elite institutional Gold (XAUUSD) price action analyst. "
        "You specialize in market structure, liquidity, order flow, and high-probability 1H candle prediction. "
        "Respond ONLY with a single valid JSON object. No markdown, no extra text."
    )

    user_prompt = f"""Asset: XAUUSD (Gold)
Timeframe: 1H
Current Price: {price}

Recent Hourly Candles (oldest → newest):
{candles}

Perform expert-level analysis:
1. Identify current market structure (HH/HL or LH/LL)
2. Note any liquidity sweeps, equal highs/lows, or order blocks
3. Assess momentum and candle body/wick behavior
4. Predict the NEXT 1H candle only

Respond with ONLY this JSON:
{{
  "direction": "bullish" or "bearish",
  "expected_high": <number>,
  "expected_low": <number>,
  "confidence": <number 0-100>,
  "reason": "<1-2 short expert sentences>"
}}"""

    last_error = None
    max_attempts = max(len(CANDIDATE_MODELS), 1) + 3

    for attempt in range(1, max_attempts + 1):
        current_model = get_active_model()
        use_json_mode = current_model not in JSON_MODE_UNSUPPORTED
        reasoning = is_reasoning_model(current_model)
        token_budget = REASONING_MODEL_MAX_TOKENS if reasoning else AI_MAX_TOKENS

        try:
            kwargs = dict(
                model=current_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.15,
                max_tokens=token_budget,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if reasoning:
                kwargs["reasoning_effort"] = "low"

            try:
                response = client.chat.completions.create(**kwargs)
            except TypeError:
                kwargs.pop("response_format", None)
                kwargs.pop("reasoning_effort", None)
                response = client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content
            if not content or not content.strip():
                last_error = f"Empty response from {current_model}"
                log(f"{symbol} attempt {attempt}: {last_error}")
                time.sleep(2)
                continue

            data = extract_json_object(content)

            required = ("direction", "expected_high", "expected_low", "reason")
            if any(k not in data for k in required):
                last_error = f"Missing fields: {data}"
                continue

            direction = str(data.get("direction", "")).lower()
            if direction not in ("bullish", "bearish"):
                last_error = f"Invalid direction: {direction}"
                continue

            return build_prediction_message(symbol, price, data)

        except Exception as e:
            err = str(e).lower()
            last_error = f"{type(e).__name__}: {e}"
            log(f"{symbol} attempt {attempt}: {last_error}")

            if "model_not_found" in err or "does not exist" in err:
                advance_model()
            elif "json_validate_failed" in err:
                JSON_MODE_UNSUPPORTED.add(current_model)
            else:
                time.sleep(2)

    return f"⚠️ XAUUSD — Analysis failed after {max_attempts} attempts.\nLast error: {last_error}"


def run_analysis_cycle():
    log("=" * 50)
    log("Starting XAUUSD analysis cycle")
    log("=" * 50)

    if is_forex_market_closed():
        msg = "💤 XAUUSD — Market Closed (Weekend)\nNext analysis when market reopens."
        send_to_ntfy(msg, title="XAUUSD - Market Closed")
        return

    price, candles = get_candles("GC=F")
    if price is None:
        send_to_ntfy(f"⚠️ XAUUSD — Data Error\n{candles}", title="XAUUSD - Data Error")
    else:
        msg = analyze("XAUUSD", price, candles)
        send_to_ntfy(msg, title="XAUUSD Signal")

    log("Cycle complete.")


def seconds_until_next_hour():
    now = datetime.now()
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return (next_hour - now).total_seconds()


def main():
    log("Taurus XAUUSD Bot starting...")

    if not API_KEY:
        log("FATAL: GOOGLE_API_KEY / GROQ_API_KEY missing")
        send_to_ntfy("❌ Bot failed: API key missing", title="Bot Error")
        return

    discover_models()
    log(f"Active model: {get_active_model()}")

    send_to_ntfy(
        f"✅ Taurus XAUUSD Bot started\nModel: {get_active_model()}",
        title="Bot Started"
    )

    try:
        run_analysis_cycle()
    except Exception as e:
        send_to_ntfy(f"⚠️ Startup cycle error: {e}", title="Bot Error")

    while True:
        wait = seconds_until_next_hour()
        log(f"Waiting {int(wait//60)}m {int(wait%60)}s until next hour...")
        time.sleep(wait)

        try:
            run_analysis_cycle()
        except Exception as e:
            send_to_ntfy(f"⚠️ Hourly cycle error: {e}", title="Bot Error")

        time.sleep(2)


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except Exception as e:
            log(f"FATAL: {e} — restarting in 30s")
            try:
                send_to_ntfy(f"❌ Bot crashed: {e}\nRestarting...", title="Bot Crash")
            except Exception:
                pass
            time.sleep(30)