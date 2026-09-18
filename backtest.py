import argparse
import os
import sys
import time

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as bot  # reuses generate_google_content / extract_json_object / MODEL_CANDIDATES / send_to_ntfy


def format_candles(df):
    lines = []
    for idx, row in df.iterrows():
        ts = idx.strftime("%m-%d %H:%M")
        lines.append(
            f"{ts} | O:{row['Open']:.2f} H:{row['High']:.2f} "
            f"L:{row['Low']:.2f} C:{row['Close']:.2f}"
        )
    return "\n".join(lines)


def get_signal(price, candles_text):
    """Mirrors main.py's analyze() prompt but returns the parsed dict
    instead of a formatted notification string. Keep this prompt in sync
    with main.py's analyze() if you change one, change the other."""
    system_prompt = (
        "You are Taurus AI — an elite institutional Gold (XAUUSD) price action analyst. "
        "Reply with ONLY a valid JSON object. No markdown, no extra text."
    )
    user_prompt = f"""Asset: XAUUSD (Gold)
Timeframe: 1H
Current Price: {price}

Recent Hourly Candles (oldest → newest):
{candles_text}

Perform expert-level technical analysis and predict the NEXT 1H candle only.
Also propose one concrete trade setup consistent with that prediction
(entry near current price or a sensible pullback level, a stop loss beyond
recent structure, and a take profit; risk_reward should be the TP distance
divided by the SL distance, e.g. "1:2").

Return JSON with keys: direction, expected_high, expected_low, confidence,
reason, entry_price, stop_loss, take_profit, risk_reward.
"""
    last_error = None
    for model_name in bot.MODEL_CANDIDATES:
        try:
            response = bot.generate_google_content(model_name, system_prompt, user_prompt)
            content = getattr(response, "text", None)
            if not content or not content.strip():
                raise ValueError("empty response from model")
            data = bot.extract_json_object(content)
            required = ("direction", "entry_price", "stop_loss", "take_profit")
            if any(k not in data for k in required):
                raise ValueError(f"missing fields: {data}")
            return data
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"all model candidates failed: {last_error}")


def simulate_trade(df, entry_idx, direction, sl, tp, lookahead_hours):
    """Walk forward from entry_idx and see whether SL or TP is hit first."""
    end_idx = min(entry_idx + lookahead_hours, len(df) - 1)
    for i in range(entry_idx, end_idx + 1):
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        if direction == "bullish":
            hit_sl = low <= sl
            hit_tp = high >= tp
        else:
            hit_sl = high >= sl
            hit_tp = low <= tp
        if hit_sl and hit_tp:
            return "SL", i  # ambiguous same-candle hit -> counted as a loss
        if hit_sl:
            return "SL", i
        if hit_tp:
            return "TP", i
    return "NO_HIT", end_idx


def run_backtest(days, window, stride, lookahead_hours, out_base):
    print(f"Fetching {days}d of GC=F hourly data from yfinance...")
    df = yf.Ticker("GC=F").history(period=f"{days}d", interval="1h")
    df = df.dropna()
    if df.empty:
        msg = f"⚠️ Taurus Backtest — no data returned by yfinance for --days {days}."
        print(msg)
        bot.send_to_ntfy(msg, title="Taurus Backtest Error")
        return
    print(f"Got {len(df)} hourly candles.\n")

    rows = []
    wins = losses = no_hits = 0
    r_multiples = []

    i = window
    while i < len(df) - 1:
        window_df = df.iloc[i - window:i]
        price = round(float(df["Close"].iloc[i - 1]), 2)
        candles_text = format_candles(window_df)
        ts = df.index[i]

        try:
            data = get_signal(price, candles_text)
        except Exception as exc:
            print(f"[{ts}] signal failed: {exc}")
            i += stride
            time.sleep(1)
            continue

        direction = str(data.get("direction", "")).strip().lower()
        try:
            entry = float(data["entry_price"])
            sl = float(data["stop_loss"])
            tp = float(data["take_profit"])
        except (KeyError, TypeError, ValueError):
            print(f"[{ts}] unusable trade levels, skipping: {data}")
            i += stride
            time.sleep(1)
            continue

        if direction not in ("bullish", "bearish"):
            print(f"[{ts}] unclear direction '{direction}', skipping")
            i += stride
            time.sleep(1)
            continue

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk == 0:
            print(f"[{ts}] zero-risk setup, skipping")
            i += stride
            time.sleep(1)
            continue

        outcome, hit_idx = simulate_trade(df, i, direction, sl, tp, lookahead_hours)

        if outcome == "TP":
            r = reward / risk
            wins += 1
        elif outcome == "SL":
            r = -1.0
            losses += 1
        else:
            exit_price = float(df["Close"].iloc[hit_idx])
            raw = (exit_price - entry) if direction == "bullish" else (entry - exit_price)
            r = raw / risk
            no_hits += 1

        r_multiples.append(r)
        rows.append({
            "timestamp": ts, "direction": direction, "price": price,
            "entry": entry, "sl": sl, "tp": tp, "outcome": outcome,
            "r_multiple": round(r, 2), "confidence": data.get("confidence"),
            "reason": str(data.get("reason", ""))[:200],
        })
        print(f"[{ts}] {direction.upper():8s} entry={entry} sl={sl} tp={tp} "
              f"-> {outcome} (R={r:+.2f})")

        i += stride
        time.sleep(1)  # be polite to API rate limits

    if not r_multiples:
        msg = f"⚠️ Taurus Backtest — no usable signals generated over {days}d (stride={stride}h)."
        print("\n" + msg)
        bot.send_to_ntfy(msg, title="Taurus Backtest — No Signals")
        return

    total = len(r_multiples)
    win_rate = wins / total * 100
    avg_r = sum(r_multiples) / total
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in r_multiples:
        cum += r
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    summary_lines = [
        "📊 **TAURUS AI — Backtest Complete**",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"Period: last {days}d | stride: {stride}h | lookahead: {lookahead_hours}h",
        f"Signals generated : {total}",
        f"Wins (TP)          : {wins}",
        f"Losses (SL)        : {losses}",
        f"No hit (timeout)   : {no_hits}",
        f"Win rate           : {win_rate:.1f}%",
        f"Avg R per trade    : {avg_r:+.2f}",
        f"Total R (sum)      : {cum:+.2f}",
        f"Max drawdown (R)   : {max_dd:.2f}",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "⚠️ Backtest only — not a guarantee of live performance.",
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    bot.send_to_ntfy(summary_text, title="Taurus Backtest Complete")

    df_results = pd.DataFrame(rows)
    csv_path = f"{out_base}.csv"
    xlsx_path = f"{out_base}.xlsx"
    df_results.to_csv(csv_path, index=False)
    df_results.to_excel(xlsx_path, index=False, sheet_name="Backtest")
    print(f"\nDetailed per-signal log saved to {csv_path} and {xlsx_path}")


def main():
    parser = argparse.ArgumentParser(description="Backtest Taurus AI XAUUSD signals")
    parser.add_argument("--days", type=int, default=30, help="How many days of hourly history to test (default 30)")
    parser.add_argument("--window", type=int, default=12, help="Candles fed to the model per signal (default 12, matches main.py)")
    parser.add_argument("--stride", type=int, default=4, help="Hours between simulated signals, to control API cost (default 4)")
    parser.add_argument("--lookahead-hours", type=int, default=24, help="Max hours to wait for SL/TP to be hit (default 24)")
    parser.add_argument("--out", default="backtest_results", help="Output file base name (without extension) — saves both .csv and .xlsx")
    args = parser.parse_args()

    if not bot.API_KEY:
        print("FATAL: GOOGLE_API_KEY is not set in your environment.")
        return

    run_backtest(args.days, args.window, args.stride, args.lookahead_hours, args.out)


if __name__ == "__main__":
    main()
