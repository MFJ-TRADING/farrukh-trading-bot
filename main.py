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
    "XAUUSD": {"yf": "GC=F", "type": "futures"},   # Gold futures - closed weekends
    "EURUSD": {"yf": "EURUSD=X", "type": "forex"}, # Forex - closed weekends
    
    "ETHUSD": {"yf": "ETH-USD", "type": "crypto"}, # Crypto - 24/7
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "mf-trading-bot-90")
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"

# Preferred order — bot picks the first one it actually has access to.
# If Groq deprecates one, it just falls through to the next automatically.
MODEL_PREFERENCE = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]

MAX_NTFY_RETRIES = 2        # retry sending notification if it fails
SLEEP_BETWEEN_SYMBOLS = 2   # seconds, avoid rate limits / overlapping notifications
AI_MAX_TOKENS = 300         # default for plain instruct models, no reasoning overhead needed
REASONING_MODEL_MAX_TOKENS = 1200  # gpt-oss / reasoning models spend tokens "thinking" before answering


def is_reasoning_model(model_name):
    """gpt-oss models (and similar) use hidden chain-of-thought tokens before
    producing visible content, so they need a much larger token budget or
    the answer comes back empty with finish_reason=length."""
    low = model_name.lower()
    return "gpt-oss" in low or "deepseek" in low or "r1" in low

client = Groq(api_key=GROQ_API_KEY)

# Populated at startup by discover_models(); analyze() rotates through this
# list automatically if a model returns "model_not_found".
CANDIDATE_MODELS = []
CURRENT_MODEL_IDX = 0

# Models that reject response_format={"type":"json_object"} at the server
# level (json_validate_failed) go here — future calls skip forcing that mode.
JSON_MODE_UNSUPPORTED = set()


def extract_json_object(text):
    """
    Robustly pull the first valid JSON object out of a text blob, even if
    the model added stray preamble/postamble text around it despite
    instructions not to. Returns a dict or raises json.JSONDecodeError.
    """
    text = text.strip()
    # Fast path: the whole thing is already valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Fallback: find the outermost { ... } block via bracket matching
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    return json.loads(candidate)  # raises if still invalid

    # Nothing worked — raise a clean error
    raise json.JSONDecodeError("No valid JSON object found in response", text, 0)



# ============================================================
# MODEL DISCOVERY — ask Groq what this account can actually use
# ============================================================
def discover_models():
    """
    Query Groq's /models endpoint to see which models this API key
    actually has access to, then build a candidate list ordered by
    preference. Falls back to the raw preference list if the query
    fails (e.g. network issue) — analyze() will still self-correct
    via model_not_found handling in that case.
    """
    global CANDIDATE_MODELS
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=15,
        )
        if resp.status_code != 200:
            log(f"Could not fetch model list ({resp.status_code}): {resp.text}")
            CANDIDATE_MODELS = list(MODEL_PREFERENCE)
            return

        available_ids = [m["id"] for m in resp.json().get("data", [])]
        log(f"Groq account has access to {len(available_ids)} models: {available_ids}")

        ordered = [m for m in MODEL_PREFERENCE if m in available_ids]

        # add any other usable chat models not in our preference list, as extra fallback
        for mid in available_ids:
            low = mid.lower()
            if mid not in ordered and not any(x in low for x in ("whisper", "guard", "tts", "compound")):
                ordered.append(mid)

        if not ordered:
            log("WARNING: none of the preferred models were found in account's model list.")
            ordered = list(MODEL_PREFERENCE)

        CANDIDATE_MODELS = ordered
        log(f"Model candidates in order of use: {CANDIDATE_MODELS}")

    except Exception as e:
        log(f"Exception discovering models: {e} — falling back to static preference list.")
        CANDIDATE_MODELS = list(MODEL_PREFERENCE)


def get_active_model():
    if not CANDIDATE_MODELS:
        return MODEL_PREFERENCE[0]
    return CANDIDATE_MODELS[CURRENT_MODEL_IDX % len(CANDIDATE_MODELS)]


def advance_model():
    global CURRENT_MODEL_IDX
    CURRENT_MODEL_IDX += 1
    log(f"Switching to next candidate model: {get_active_model()}")



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
# AI ANALYSIS — strict JSON output, parsed and validated in Python
# ============================================================
def build_prediction_message(symbol, price, data):
    """Build the final human-readable notification text from parsed JSON fields."""
    direction = str(data.get("direction", "unknown")).strip().lower()
    emoji = "🟢" if direction == "bullish" else ("🔴" if direction == "bearish" else "⚪")

    expected_high = data.get("expected_high", "N/A")
    expected_low = data.get("expected_low", "N/A")
    reason = str(data.get("reason", "No reason provided.")).strip()
    confidence = data.get("confidence", None)

    lines = [
        f"📊 {symbol} — Next 1H Candle Prediction",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Current Price: {price}",
        f"{emoji} Next Candle: {direction.capitalize()}",
        f"📈 Expected High: {expected_high}",
        f"📉 Expected Low: {expected_low}",
    ]
    if confidence is not None:
        lines.append(f"🎯 Confidence: {confidence}")
    lines.append(f"Reason: {reason}")

    return "\n".join(lines)


def analyze(symbol, price, candles):
    system_prompt = (
        "You are a professional price action trading analyst. "
        "You respond ONLY with a single valid JSON object — no markdown, "
        "no code fences, no explanation text outside the JSON. "
        "Base your prediction strictly on the candle data provided. "
        "Prioritize accuracy over confidence: if the pattern is unclear, "
        "say so honestly in the reason and lower the confidence score."
    )

    user_prompt = f"""Asset: {symbol}
Current Price: {price}
Last 10 Hourly Candles (oldest to newest):
{candles}

Analyze the price action (trend, momentum, support/resistance from highs/lows,
candle body/wick patterns) and predict the NEXT 1-hour candle.

Respond with ONLY this exact JSON structure, no other text:
{{
  "direction": "bullish" or "bearish",
  "expected_high": <number>,
  "expected_low": <number>,
  "confidence": <number 0-100>,
  "reason": "<one short sentence explaining the price-action reasoning>"
}}"""

    last_error = None
    max_attempts = max(len(CANDIDATE_MODELS), 1) + 3  # room to rotate models + json_mode fallback + retries

    for attempt in range(1, max_attempts + 1):
        current_model = get_active_model()
        use_json_mode = current_model not in JSON_MODE_UNSUPPORTED
        reasoning_model = is_reasoning_model(current_model)
        token_budget = REASONING_MODEL_MAX_TOKENS if reasoning_model else AI_MAX_TOKENS

        try:
            request_kwargs = dict(
                model=current_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=token_budget,
            )
            if use_json_mode:
                request_kwargs["response_format"] = {"type": "json_object"}
            if reasoning_model:
                request_kwargs["reasoning_effort"] = "low"

            try:
                response = client.chat.completions.create(**request_kwargs)
            except TypeError:
                # older groq SDK versions may not accept reasoning_effort/response_format
                request_kwargs.pop("response_format", None)
                request_kwargs.pop("reasoning_effort", None)
                response = client.chat.completions.create(**request_kwargs)

            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            if not content or not content.strip():
                last_error = f"Empty content (finish_reason={finish_reason}) [model={current_model}, max_tokens={token_budget}]"
                log(f"{symbol} AI attempt {attempt}: {last_error}")
                time.sleep(2)
                continue

            try:
                data = extract_json_object(content)
            except json.JSONDecodeError as je:
                last_error = f"Invalid JSON: {je} | raw: {content.strip()[:200]} [model={current_model}]"
                log(f"{symbol} AI attempt {attempt}: {last_error}")
                time.sleep(2)
                continue

            # Validate required fields
            required = ("direction", "expected_high", "expected_low", "reason")
            missing = [k for k in required if k not in data]
            if missing:
                last_error = f"Missing fields in JSON: {missing} | raw: {data} [model={current_model}]"
                log(f"{symbol} AI attempt {attempt}: {last_error}")
                time.sleep(2)
                continue

            if str(data.get("direction", "")).strip().lower() not in ("bullish", "bearish"):
                last_error = f"Invalid direction value: {data.get('direction')} [model={current_model}]"
                log(f"{symbol} AI attempt {attempt}: {last_error}")
                time.sleep(2)
                continue

            # Success — build the final message
            return build_prediction_message(symbol, price, data)

        except Exception as e:
            err_text = str(e).lower()
            last_error = f"{type(e).__name__}: {e} [model={current_model}, json_mode={use_json_mode}]"
            log(f"{symbol} AI attempt {attempt} exception: {last_error}")

            if "model_not_found" in err_text or ("does not exist" in err_text and "model" in err_text):
                # Model itself is inaccessible — permanently rotate away from it.
                advance_model()
            elif "json_validate_failed" in err_text or "failed to validate json" in err_text:
                # This model's server-side JSON-mode validator is unreliable —
                # stop forcing json_object mode for it and rely on prompt +
                # robust extraction instead. Retry immediately, same model.
                JSON_MODE_UNSUPPORTED.add(current_model)
                log(f"Disabling forced JSON mode for {current_model}, retrying with plain prompt.")
            else:
                time.sleep(2)

    return (
        f"⚠️ {symbol} — AI analysis failed after {max_attempts} attempts.\n"
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

    # 1. Discover which models this account actually has access to
    discover_models()
    active_model = get_active_model()
    log(f"Using model: {active_model}")

    # 2. Startup test notification
    send_to_ntfy(
        f"✅ Bot started successfully.\nUsing AI model: {active_model}\n"
        f"Running last hour's analysis now...",
        title="Bot Test Notification",
    )

    # 3. Immediate analysis using the most recent completed hourly candle
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
            
