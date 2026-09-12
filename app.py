import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Environment Variables
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def send_discord_embed(title, description, color=3447003, fields=[]):
    """Sends a styled embed message to Discord"""
    if not DISCORD_WEBHOOK_URL:
        return
        
    embed = {
        "title": title,
        "description": description,
        "color": color,
        "fields": fields,
        "footer": {"text": "Tauric AI Multi-Agent System"}
    }
    
    payload = {"embeds": [embed]}
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload)
    except Exception as e:
        print(f"Error sending to Discord: {e}")

def get_ai_analysis(symbol, action, timeframe, price):
    prompt = f"""
    You are an AI Trading Agent system analyzing a signal for {symbol}.
    Signal Type: {action}
    Timeframe: {timeframe}
    Current Price: {price}

    Conduct a multi-agent analysis:
    1. Technical Bias Analysis
    2. Risk Management & Position Sizing Warning
    3. Final Verdict (CONFIRM / REJECT / WAIT) with concise reasoning.
    
    Provide output in clear, structured Roman Urdu/English for mobile notifications.
    Keep it concise and actionable.
    """
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"AI Analysis Error: {response.text}"
    except Exception as e:
        return f"Error contacting Groq API: {str(e)}"

@app.route('/', methods=['GET'])
def home():
    return "AI Trading Discord Server is Live!", 200

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload"}), 400

        symbol = data.get("symbol", "UNKNOWN")
        action = str(data.get("action", "ALERT")).upper()
        timeframe = data.get("timeframe", "15m")
        price = data.get("price", "N/A")

        # Color: Green for BUY, Red for SELL, Blue for default
        embed_color = 5763719 if "BUY" in action else (15548927 if "SELL" in action else 3447003)

        # 1. Immediate Signal Notification
        signal_fields = [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Action", "value": action, "inline": True},
            {"name": "Price", "value": str(price), "inline": True},
            {"name": "Timeframe", "value": timeframe, "inline": True}
        ]
        send_discord_embed("🚨 TradingView Signal Received", "⏳ *AI Agents are running multi-agent analysis...*", color=embed_color, fields=signal_fields)

        # 2. Run AI Analysis
        ai_result = get_ai_analysis(symbol, action, timeframe, price)

        # 3. Final AI Report Notification
        send_discord_embed(f"📊 AI Analysis Report - {symbol}", ai_result, color=embed_color)

        return jsonify({"status": "success"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
