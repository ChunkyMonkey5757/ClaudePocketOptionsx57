# PocketBotX57 — AI-Powered Trade Signal Engine

PocketBotX57 is a fully autonomous, AI-enhanced trading signal bot designed for rapid-fire decision-making on platforms like **Pocket Option**.

## Features

- **Real-Time Signal Engine** using RSI, MACD, Bollinger Bands, and more
- **Telegram Bot Interface** with commands:
  - `/won`, `/lost` — Track trade outcomes
  - `/summary` — View performance metrics
  - `/weights` — See adaptive strategy weights
  - `/history` — Recent trade signals
- **Modular Strategy System** with adaptive AI learning
- **Multi-Source Price Feeds** from CoinGecko, CoinMarketCap, Alpha Vantage, etc.
- **Self-Tuning Weights** based on historical performance
- **Built-in Logger, Watchdog, and Signal Journal**

---

## Project Structure

```
src/
├── core/               # Startup + runtime logic
├── telegram_bot/       # Bot controller + commands
├── strategies/         # Trading strategy modules
├── api_handler/        # Exchange + data API adapters
├── ai_learning/        # AI feedback, performance stats, adaptive weights
├── utils/              # Logging, math, and formatting helpers
├── storage/            # Trade logs, journals
.env                    # Your real config (edit this)
requirements.txt        # Install dependencies
start.sh                # Run this on Glitch or locally
```

---

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Edit `.env`**
   Add your Telegram token, Kraken keys, API keys, and admin user ID.

3. **Run the Bot**
   ```bash
   bash start.sh
   ```

---

## Deployment Options

- **Local:** Python 3.10+ required
- **Glitch / Vercel:** Use `start.sh` or import from GitHub
- **GitHub:** Push the full repo and sync with CI/CD if desired

---

## License

MIT — Use, modify, and deploy freely.
