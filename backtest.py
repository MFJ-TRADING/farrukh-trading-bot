import os
import time
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import yfinance as yf
import pandas as pd

try:
    from google import genai as google_genai_module
except Exception:  # pragma: no cover - dependency may not be installed yet
    google_genai_module = None

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - dependency may not be installed yet
    genai = None

# ============================================================
# CONFIG
# ============================================================
SYMBOLS = {"XAUUSD": {"yf": "GC=F", "type": "futures"}}
DXY_SYMBOL = "DX-Y.NYB"  # US Dollar Index - gold is usually inversely correlated
API_KEY = os.environ.get("GOOGLE_API_KEY")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "mf-trading-bot-90")
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK")

# NOTE: gemini-1.5-flash and gemini-1.5-flash-8b have been fully shut down by
# Google (retired in 2025) and now return a 404 "model not found" error on
# every call. "gemini-flash-latest" is Google's alias that always points at
# the newest stable Flash model, so it's the safest first choice.
MODEL_CANDIDATES = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
]
MAX_NTFY_RETRIES = 1
MAX_DISCORD_RETRIES = 3
AI_MAX_TOKENS = 4096

# Discord alerts only 9 AM - 9 PM Pakistan time (assumes Asia/Karachi; the
# job itself still runs hourly around the clock for ntfy, this only gates
# the Discord send). Change the timezone here if this assumption is wrong.
DISCORD_ALERT_TZ = ZoneInfo("Asia/Karachi")
DISCORD_ALERT_START_HOUR = 9   # 9:00 AM local
DISCORD_ALERT_END_HOUR = 21    # 9:00 PM local (window is [9, 21))


def is_within_discord_alert_window():
    now_local = datetime.now(DISCORD_ALERT_TZ)
    return DISCORD_ALERT_START_HOUR <= now_local.hour < DISCORD_ALERT_END_HOUR


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def get_google_model(model_name=None):
    if not API_KEY:
        raise RuntimeError("Missing GOOGLE_API_KEY")

    chosen = model_name or MODEL_CANDIDATES[0]

    if google_genai_module is not None:
        return {
            "provider": "google-genai",
            "model": chosen,
            "client": google_genai_module.Client(api_key=API_KEY),
        }

    if genai is None:
        raise RuntimeError(
            "Gemini SDK is not installed. Install 'google-genai' or 'google-generativeai'."
        )

    genai.configure(api_key=API_KEY)
    return {
        "provider": "legacy-google-generativeai",
        "model": chosen,
        "client": genai.GenerativeModel(chosen),
    }


def generate_google_content(model_name, system_prompt, user_prompt):
    model_info = get_google_model(model_name)

    if model_info["provider"] == "google-genai":
        try:
            from google.genai import types
        except Exception as exc:  # pragma: no cover - dependency may not be installed yet
            raise RuntimeError("google-genai SDK is missing types support") from exc

        # Gemini 2.5 models "think" by default, which can silently eat the
        # entire max_output_tokens budget on internal reasoning and leave
        # response.text empty or truncated. Disable thinking for 2.5 models;
        # Gemini 3 models use a different knob (thinking_level) and error
        # out if thinking_budget is also set.
        if "gemini-3" in model_info["model"]:
            thinking_config = types.ThinkingConfig(thinking_level="low")
        else:
            thinking_config = types.ThinkingConfig(thinking_budget=0)

        return model_info["client"].models.generate_content(
            model=model_info["model"],
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.15,
                max_output_tokens=AI_MAX_TOKENS,
                response_mime_type="application/json",
                thinking_config=thinking_config,
            ),
        )

    return model_info["client"].generate_content(
        [system_prompt, user_prompt],
        generation_config={
            "temperature": 0.15,
            "max_output_tokens": AI_MAX_TOKENS,
            "response_mime_type": "application/json",
        },
    )


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


def send_to_ntfy(message, title="XAUUSD Signal"):
    if not NTFY_TOPIC:
        log("ntfy SKIPPED: NTFY_TOPIC not set")
        return False
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
                log(f"ntfy OK ({title}) -> topic '{NTFY_TOPIC}'")
                return True
            log(f"ntfy attempt {attempt} failed: {response.status_code} {response.text[:200]}")
        except Exception as e:
            log(f"ntfy attempt {attempt} exception: {e}")
        if attempt < MAX_NTFY_RETRIES:
            time.sleep(3)
    return False


def send_to_discord(message, title="XAUUSD Signal"):
    if not DISCORD_WEBHOOK:
        log("discord SKIPPED: DISCORD_WEBHOOK not set")
        return False

    content = f"**{title}**\n{message}"
    # Discord hard-caps message content at 2000 chars
    if len(content) > 1990:
        content = content[:1987] + "..."

    for attempt in range(1, MAX_DISCORD_RETRIES + 1):
        try:
            response = requests.post(
                DISCORD_WEBHOOK,
                json={"content": content},
                timeout=15,
            )
            # Discord webhooks return 204 No Content on success
            if response.status_code in (200, 204):
                log(f"discord OK ({title})")
                return True
            log(f"discord attempt {attempt} failed: {response.status_code} {response.text[:200]}")
        except Exception as e:
            log(f"discord attempt {attempt} exception: {e}")
        if attempt < MAX_DISCORD_RETRIES:
            time.sleep(3)
    return False


def notify_all(message, title="XAUUSD Signal"):
    ntfy_ok = send_to_ntfy(message, title=title)

    if is_within_discord_alert_window():
        discord_ok = send_to_discord(message, title=title)
    else:
        log("discord SKIPPED: outside 9 AM-9 PM alert window (Asia/Karachi)")
        discord_ok = False

    return ntfy_ok, discord_ok


# ============================================================
# TECHNICAL INDICATORS
# ============================================================
# LLMs are unreliable at doing arithmetic over a raw list of OHLC numbers
# in their head. Pre-computing the standard indicators here and handing the
# model the *results* (not just raw candles) is the single biggest lever
# for getting a more grounded, less hallucinated analysis.

def compute_indicators(df):
    """Given an OHLCV dataframe, return the latest value of each indicator."""
    close, high, low = df["Close"], df["High"], df["Low"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi14 = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr14 = true_range.rolling(14).mean()

    bb_mid = sma20
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    def last(series):
        s = series.dropna()
        return round(float(s.iloc[-1]), 2) if len(s) else None

    return {
        "sma20": last(sma20),
        "sma50": last(sma50),
        "rsi14": last(rsi14),
        "macd": last(macd_line),
        "macd_signal": last(macd_signal),
        "macd_hist": last(macd_hist),
        "atr14": last(atr14),
        "bb_upper": last(bb_upper),
        "bb_lower": last(bb_lower),
    }


def find_support_resistance(df, lookback=50):
    recent = df.tail(lookback)
    if recent.empty:
        return None, None
    return round(float(recent["Low"].min()), 2), round(float(recent["High"].max()), 2)


def format_indicators(ind):
    parts = []
    for key, val in ind.items():
        if val is not None:
            parts.append(f"{key}={val}")
    return ", ".join(parts) if parts else "insufficient history for indicators"


def get_dxy_context():
    """US Dollar Index, best-effort - gold and DXY are usually inversely
    correlated, so this gives the model useful cross-asset context. Never
    fails the whole cycle if unavailable."""
    try:
        df = yf.Ticker(DXY_SYMBOL).history(period="5d", interval="1d")
        if df.empty:
            return None
        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
        change_pct = ((last_close - prev_close) / prev_close * 100) if prev_close else 0.0
        return {"value": round(last_close, 2), "change_pct": round(change_pct, 2)}
    except Exception as e:
        log(f"DXY fetch failed (non-fatal): {e}")
        return None


def get_market_data(yf_symbol, hourly_n=24):
    """Fetches 1H, 4H (resampled) and Daily data plus indicators/support-
    resistance for all three, so the model gets both a close-up and a
    big-picture view instead of judging 1H candles in isolation."""
    try:
        ticker = yf.Ticker(yf_symbol)
        hourly_df = ticker.history(period="10d", interval="1h")
        if hourly_df.empty:
            return None, "No hourly data returned"

        daily_df = ticker.history(period="6mo", interval="1d")

        price = round(float(hourly_df["Close"].iloc[-1]), 2)

        # 1H candle listing (with volume, which was previously missing)
        hourly_recent = hourly_df.tail(hourly_n)
        hourly_lines = []
        for idx, row in hourly_recent.iterrows():
            ts = idx.strftime("%m-%d %H:%M")
            hourly_lines.append(
                f"{ts} | O:{row['Open']:.2f} H:{row['High']:.2f} "
                f"L:{row['Low']:.2f} C:{row['Close']:.2f} V:{row['Volume']:.0f}"
            )

        # 4H view via resampling the 1H data
        four_h_df = (
            hourly_df.resample("4h")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna()
        )

        support, resistance = find_support_resistance(hourly_df, lookback=50)

        result = {
            "price": price,
            "hourly_candles": "\n".join(hourly_lines),
            "hourly_indicators": format_indicators(compute_indicators(hourly_df)),
            "four_h_indicators": format_indicators(compute_indicators(four_h_df)) if len(four_h_df) >= 20 else "insufficient 4H history",
            "daily_indicators": format_indicators(compute_indicators(daily_df)) if not daily_df.empty and len(daily_df) >= 20 else "insufficient daily history",
            "support": support,
            "resistance": resistance,
            "dxy": get_dxy_context(),
        }
        return result, None
    except Exception as e:
        return None, str(e)


def build_prediction_message(price, data):
    direction = str(data.get("direction", "unknown")).strip().lower()
    bias_label = "BULLISH" if direction == "bullish" else ("BEARISH" if direction == "bearish" else "NEUTRAL")
    indicator = "🟢" if direction == "bullish" else ("🔴" if direction == "bearish" else "⚪")
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    confidence = data.get("confidence", "N/A")
    confidence_str = f"{confidence}%" if confidence != "N/A" else "N/A"

    lines = [
        f"{indicator}  **TAURUS AI — XAUUSD SIGNAL**",
        f"1H Outlook · {timestamp}",
        "",
        f"**Price:** {price}   |   **Bias:** {bias_label}   |   **Confidence:** {confidence_str}",
        f"**Expected Range:** {data.get('expected_low', 'N/A')} – {data.get('expected_high', 'N/A')}",
        "",
        "**Trade Setup**",
        f"Entry — {data.get('entry_price', 'N/A')}",
        f"Stop Loss — {data.get('stop_loss', 'N/A')}",
        f"Take Profit — {data.get('take_profit', 'N/A')}",
        f"Risk:Reward — {data.get('risk_reward', 'N/A')}",
        "",
        "**Analysis**",
        data.get('reason', 'No reason provided.'),
        "",
        "_Not financial advice. Manage your risk._",
    ]
    return "\n".join(lines)


def analyze(symbol, market_data):
    price = market_data["price"]
    dxy = market_data["dxy"]
    dxy_line = (
        f"US Dollar Index (DXY): {dxy['value']} ({dxy['change_pct']:+.2f}% vs prior day) "
        f"- gold is typically inversely correlated with DXY"
        if dxy else "US Dollar Index: unavailable"
    )

    system_prompt = (
        "You are Taurus AI, an elite institutional Gold (XAUUSD) price action "
        "analyst. You combine multiple timeframes and indicators rather than "
        "reacting to short-term noise. Only assign high confidence when "
        "multiple signals (trend, momentum, volatility, multi-timeframe "
        "alignment) genuinely agree - if signals conflict, say so and lower "
        "confidence accordingly. Reply with ONLY a valid JSON object, no "
        "markdown, no extra text."
    )

    user_prompt = f"""Asset: XAUUSD (Gold)
Current Price: {price}

=== 1H Candles (oldest -> newest) ===
{market_data['hourly_candles']}

=== 1H Indicators ===
{market_data['hourly_indicators']}

=== 4H Indicators (broader trend context) ===
{market_data['four_h_indicators']}

=== Daily Indicators (macro trend context) ===
{market_data['daily_indicators']}

=== Key Levels ===
Recent Support: {market_data['support']}
Recent Resistance: {market_data['resistance']}

=== Cross-Asset Context ===
{dxy_line}

Perform expert-level multi-timeframe technical analysis and predict the
NEXT 1H candle. Weigh confluence across timeframes and indicators (trend
direction on 1H/4H/Daily, RSI overbought/oversold, MACD momentum, position
relative to Bollinger Bands, proximity to support/resistance, and DXY
context) rather than any single signal in isolation. Be conservative with
confidence when signals disagree.

Also propose one concrete trade setup consistent with that prediction
(entry near current price or a sensible pullback level, a stop loss beyond
recent structure/support-resistance, and a take profit; risk_reward should
be the TP distance divided by the SL distance, e.g. "1:2").

Return JSON with keys: direction, expected_high, expected_low, confidence,
reason, entry_price, stop_loss, take_profit, risk_reward.
"""

    last_error = None

    for attempt, model_name in enumerate(MODEL_CANDIDATES, start=1):
        try:
            response = generate_google_content(model_name, system_prompt, user_prompt)

            content = getattr(response, "text", None)
            finish_reason = None
            try:
                finish_reason = response.candidates[0].finish_reason
            except Exception:
                pass

            if not content or not content.strip():
                raise ValueError(f"Empty response from Google API (finish_reason={finish_reason})")

            try:
                data = extract_json_object(content)
            except json.JSONDecodeError:
                snippet = content[:400].replace("\n", " ")
                log(
                    f"{symbol} attempt {attempt} JSON parse failed "
                    f"(finish_reason={finish_reason}, len={len(content)}): {snippet}"
                )
                raise

            required = (
                "direction", "expected_high", "expected_low", "reason",
                "entry_price", "stop_loss", "take_profit", "risk_reward",
            )
            if any(key not in data for key in required):
                raise ValueError(f"Missing fields: {data}")

            direction = str(data.get("direction", "")).lower()
            if direction not in ("bullish", "bearish"):
                raise ValueError(f"Invalid direction: {direction}")

            return build_prediction_message(price, data)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log(f"{symbol} attempt {attempt}/{len(MODEL_CANDIDATES)} failed: {last_error}")
            if attempt < len(MODEL_CANDIDATES):
                time.sleep(2)
                continue

    return f"⚠️ XAUUSD — Analysis failed after {len(MODEL_CANDIDATES)} attempts.\nLast error: {last_error}"


def run_analysis_cycle():
    log("=" * 50)
    log("Starting XAUUSD analysis cycle")
    log("=" * 50)

    if not API_KEY:
        log("FATAL: GOOGLE_API_KEY missing")
        notify_all("⚠️ Bot error: GOOGLE_API_KEY missing", title="Bot Error")
        return False

    market_data, error = get_market_data("GC=F")
    if market_data is None:
        notify_all(f"⚠️ XAUUSD — Data Error\n{error}", title="XAUUSD - Data Error")
        return False

    msg = analyze("XAUUSD", market_data)
    ntfy_ok, discord_ok = notify_all(msg, title="XAUUSD Signal")
    log(f"Cycle complete. ntfy_ok={ntfy_ok} discord_ok={discord_ok}")
    return ntfy_ok or discord_ok


# Entry point expected by handler.py (Vercel/serverless-style handler does:
# `from main import run_once`)
def run_once():
    try:
        return run_analysis_cycle()
    except Exception as exc:
        log(f"run_once error: {exc}")
        notify_all(f"⚠️ Bot error: {exc}", title="Bot Error")
        raise


def main():
    log("Taurus XAUUSD Bot starting...")

    if not API_KEY:
        log("FATAL: GOOGLE_API_KEY missing")
        return

    try:
        run_analysis_cycle()
    except Exception as exc:
        send_to_ntfy(f"⚠️ Bot error: {exc}", title="Bot Error")
        send_to_discord(f"⚠️ Bot error: {exc}", title="Bot Error")


if __name__ == "__main__":
    main()
