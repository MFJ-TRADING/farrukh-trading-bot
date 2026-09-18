# Farrukh Trading Bot 🤖

A market analysis bot powered by the **Google Gemini API** that checks XAUUSD and sends AI-generated signals to Discord and ntfy.

## Features
- 🎯 **Manual or scheduled execution** from a local environment or CI runner
- 🤖 **AI-Powered** - Uses Google Gemini for market analysis
- 📊 **Gold-focused** - Tracks XAUUSD with hourly candles
- 💬 **Discord + ntfy Integration** - Real-time signal delivery

## Environment Variables
```bash
export GOOGLE_API_KEY=your_google_api_key
export NTFY_TOPIC=your_ntfy_topic  # optional
export DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
```

> The app now uses the current Google Gemini SDK (`google-genai`). If your key is missing or invalid, the bot will log a clear runtime error instead of silently failing.

## Run locally
```bash
python main.py
```

---
**Built with ❤️ by Farrukh**
