import os
import json
import requests
import yfinance as yf
from flask import Flask, request, jsonify
from groq import Groq

app = Flask(__name__)

# Environment Variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def get_exness_market_data(symbol="XAUUSD=X"):
    """
    Exness Gold / Forex 1-Hour Chart Data Fetcher
    Yahoo Finance Symbol for Gold is 'XAUUSD=X'
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="5d", interval="1h")
    
    current_price = round(df['Close'].iloc[-1], 2)
    # Last 10 Hourly Candles Data (OHLCV)
    recent_candles = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).to_string()
    
    return current_price, recent_candles

def analyze_with_groq_ai(symbol, price, candles_data, action=None, timeframe="1H"):
    prompt = f"""
    You are an expert Institutional Financial Analyst and Forex Trader.
    Analyze the following chart data for {symbol} (Exness Rates):
    
    Current Price: {price}
    Timeframe: {timeframe}
    Signal Action Triggered: {action if action else 'Automated Hourly Check'}
    
    Recent Hourly Candles Data (OHLCV):
    {candles_data}

    Instructions:
    1. Perform Price Action, Market Structure, and Momentum analysis.
    2. Give a clear Decision: [BUY], [SELL], or [NO TRADE / WAIT].
    3. Provide precise Entry Price, Stop Loss (SL), and Take Profit (TP1 & TP2) levels.
    4. Keep the report concise, highly accurate, and formatted cleanly for Discord markdown.
    """

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

def send_to_discord(report_text, title="🚨 AUTOMATED AI TRADE ANALYSIS 🚨"):
    if not DISCORD_WEBHOOK_URL:
        return
    payload = {"content": f"**{title}**\n\n{report_text}"}
    requests.post(DISCORD_WEBHOOK_URL, json=payload)

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

# Main Root Endpoint
@app.route('/', methods=['GET', 'POST'])
def home():
    return "Farrukh AI Trading Bot is Live & Operational!", 200

# 1. Manual Webhook Endpoint (For External POST Signals)
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        return jsonify({"status": "Active", "message": "Webhook endpoint is live. Send POST request with signal payload."}), 200

    try:
        data = request.get_json(silent=True) or {}
        symbol = data.get('symbol', 'XAUUSD')
        action = data.get('action', 'SIGNAL')
        price = data.get('price', 'N/A')
        timeframe = data.get('timeframe', '1H')

        curr_price, candles = get_exness_market_data("XAUUSD=X")
        
        ai_report = analyze_with_groq_ai(symbol, curr_price, candles, action=action, timeframe=timeframe)
        send_to_discord(ai_report, title=f"🚨 SIGNAL ALERT: {symbol} ({action}) 🚨")
        
        return jsonify({"status": "Success", "message": "Signal analyzed and sent to Discord"}), 200
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

# 2. Vercel Cron & Manual Browser Test Endpoint
@app.route('/api/cron', methods=['GET', 'POST'])
def auto_hourly_cron():
    try:
        curr_price, candles = get_exness_market_data("XAUUSD=X")
        ai_report = analyze_with_groq_ai("XAUUSD (Gold)", curr_price, candles, timeframe="1H")
        send_to_discord(ai_report, title="⏰ 1-HOUR HOURLY EXNESS AI ANALYSIS ⏰")
        
        return jsonify({"status": "Success", "message": "Hourly analysis sent to Discord"}), 200
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
