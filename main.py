import os
import time
import json
import re
from datetime import datetime

import requests
import yfinance as yf

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
API_KEY = os.environ.get("GOOGLE_API_KEY")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "mf-trading-bot-90")
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK")

# NOTE: gemini-1.5-flash and gemini-1.5-flash-8b have been fully shut down by
# Google (retired in 2025) and now return a 404 "model not found" error on
# every call. That was very likely why the whole chain of candidates failed.
# "gemini-flash-latest" is Google's alias that always points at the newest
# stable Flash model, so it's the safest first choice.
MODEL_CANDIDATES = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
]
MAX_NTFY_RETRIES = 2
MAX_DISCORD_RETRIES = 2
AI_MAX_TOKENS = 1024


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
        # response.text empty (-> JSON parse fails on an empty string).
        # Disable thinking for 2.5 models; Gemini 3 models use a different
        # knob (thinking_level) and error out if thinking_budget is also set.
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
    discord_ok = send_to_discord(message, title=title)
    return ntfy_ok, discord_ok


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


def build_prediction_message(price, data):
    direction = str(data.get("direction", "unknown")).strip().lower()
    emoji = "🟢" if direction == "bullish" else ("🔴" if direction == "bearish" else "⚪")

    lines = [
        "📊 **TAURUS AI — XAUUSD Next 1H Prediction**",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"**Current Price:** {price}",
        f"{emoji} **Next Candle:** {direction.upper()}",
        f"📈 **Expected High:** {data.get('expected_high', 'N/A')}",
        f"📉 **Expected Low:** {data.get('expected_low', 'N/A')}",
        f"🎯 **Confidence:** {data.get('confidence', 'N/A')}%",
        f"**Reason:** {data.get('reason', 'No reason provided.')}",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "💰 **Trade Setup**",
        f"▶️ **Entry:** {data.get('entry_price', 'N/A')}",
        f"🛑 **Stop Loss:** {data.get('stop_loss', 'N/A')}",
        f"🎯 **Take Profit:** {data.get('take_profit', 'N/A')}",
        f"⚖️ **Risk:Reward:** {data.get('risk_reward', 'N/A')}",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "⚠️ *Not financial advice. Manage risk.*",
    ]
    return "\n".join(lines)


def analyze(symbol, price, candles):
    system_prompt = (
        "You are Taurus AI — an elite institutional Gold (XAUUSD) price action analyst. "
        "Reply with ONLY a valid JSON object. No markdown, no extra text."
    )

    user_prompt = f"""Asset: XAUUSD (Gold)
Timeframe: 1H
Current Price: {price}

Recent Hourly Candles (oldest → newest):
{candles}

Perform expert-level technical analysis and predict the NEXT 1H candle only.
Also propose one concrete trade setup consistent with that prediction
(entry near current price or a sensible pullback level, a stop loss beyond
recent structure, and a take profit; risk_reward should be the TP distance
divided by the SL distance, e.g. "1:2").

Return JSON with keys: direction, expected_high, expected_low, confidence,
reason, entry_price, stop_loss, take_profit, risk_reward.
"""

    last_error = None

    for attempt, model_name in enumerate(MODEL_CANDIDATES, start=1):
        try:
            response = generate_google_content(model_name, system_prompt, user_prompt)

            content = getattr(response, "text", None)
            if not content or not content.strip():
                finish_reason = None
                try:
                    finish_reason = response.candidates[0].finish_reason
                except Exception:
                    pass
                raise ValueError(f"Empty response from Google API (finish_reason={finish_reason})")

            data = extract_json_object(content)
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

    price, candles = get_candles("GC=F")
    if price is None:
        notify_all(f"⚠️ XAUUSD — Data Error\n{candles}", title="XAUUSD - Data Error")
        return False

    msg = analyze("XAUUSD", price, candles)
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
