
"""
Global Configuration Module for PocketBotX57.
Manages all runtime environment variables, strategy weights, and system defaults.
"""

import os
from dotenv import load_dotenv
from typing import Dict

# Load .env file if present
load_dotenv()

class Settings:
    """
    Centralized app configuration for all services and modules.
    """

    def __init__(self):
        # === Telegram ===
        self.BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
        self.ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "123456789"))

        # === Kraken / Exchange API Keys ===
        self.KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
        self.KRAKEN_SECRET_KEY = os.getenv("KRAKEN_SECRET_KEY", "")

        # === CoinMarketCap / Forecaster APIs ===
        self.COINMARKETCAP_API = os.getenv("CMC_API_KEY", "")
        self.FORECASTER_API_KEY = os.getenv("FORECASTER_API_KEY", "")

        # === Sentiment / NLP API (Optional) ===
        self.SENTIMENT_PROVIDER = os.getenv("SENTIMENT_PROVIDER", "textblob")

        # === Watchdog & Cooldown Settings ===
        self.WATCHDOG_TIMEOUT_SECONDS = int(os.getenv("WATCHDOG_TIMEOUT", "120"))
        self.SIGNAL_SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL", "60"))
        self.SIGNAL_COOLDOWN_SECONDS = int(os.getenv("SIGNAL_COOLDOWN", "120"))

        # === Asset Defaults ===
        self.DEFAULT_ASSETS = os.getenv("DEFAULT_ASSETS", "BTC,ETH,SOL").split(",")

        # === Strategy Weights ===
        self.STRATEGY_WEIGHTS = self._load_strategy_weights()

        # === Debug Mode ===
        self.DEBUG_MODE = bool(int(os.getenv("DEBUG_MODE", "1")))

    def _load_strategy_weights(self) -> Dict[str, float]:
        """
        Pull individual strategy weights from environment or fallback.
        """
        return {
            "rsi_strategy": float(os.getenv("WEIGHT_RSI", 0.15)),
            "macd_strategy": float(os.getenv("WEIGHT_MACD", 0.15)),
            "bollinger_strategy": float(os.getenv("WEIGHT_BOLLINGER", 0.10)),
            "sma_cross_strategy": float(os.getenv("WEIGHT_SMA", 0.10)),
            "volume_strategy": float(os.getenv("WEIGHT_VOLUME", 0.10)),
            "vwap_strategy": float(os.getenv("WEIGHT_VWAP", 0.10)),
            "pattern_strategy": float(os.getenv("WEIGHT_PATTERN", 0.10)),
            "sentiment": float(os.getenv("WEIGHT_SENTIMENT", 0.10)),
            "correlation_strategy": float(os.getenv("WEIGHT_CORRELATION", 0.10)),
            "ai_overlay": 1.0  # Final decision overlay (always active)
        }

# Singleton settings object
settings = Settings()
