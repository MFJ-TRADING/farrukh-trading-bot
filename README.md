# Farrukh Trading Bot 🤖

A Vercel serverless trading bot powered by **Groq AI** (Taurus) that analyzes financial markets hourly and sends signals to Discord.

## Features
- 🎯 **Automated Hourly Analysis** - Runs every hour via Vercel Crons
- 🤖 **AI-Powered** - Uses Groq's LLaMA model for market analysis
- 📊 **Multi-Asset** - Tracks XAUUSD, EURUSD, GBPUSD, BOOM1000, CRASH1000
- 💬 **Discord Integration** - Real-time signals to your Discord server

## Environment Variables
```
DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
GROQ_API_KEY=gsk_...
DERIV_APP_ID=1234  # (Optional) For Deriv synthetic indices
```

## Deployment
Deploy to Vercel:
```bash
vercel deploy
```

The bot will automatically run the `/api/cron` endpoint every hour.

---
**Built with ❤️ by Farrukh**
