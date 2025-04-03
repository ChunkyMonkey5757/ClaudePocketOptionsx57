"""
Configuration module for PocketBotX57.
Loads environment variables and provides configuration settings.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "")
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY", "")

# Exchange API Keys (Optional - for trading integration)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Database Configuration (for Railway deployment)
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Application Settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "120"))
MIN_CONFIDENCE_THRESHOLD = int(os.getenv("MIN_CONFIDENCE_THRESHOLD", "75"))
DEFAULT_ASSETS = os.getenv("DEFAULT_ASSETS", "BTC,ETH,BNB").split(",")

# API Configuration
API_RETRY_COUNT = 3
API_TIMEOUT_SECONDS = 30

# Signal Engine Configuration
SIGNAL_STRATEGIES = {
    "rsi_strategy": {"weight": 0.15, "enabled": True},
    "macd_strategy": {"weight": 0.20, "enabled": True},
    "bollinger_strategy": {"weight": 0.15, "enabled": True},
    "vwap_strategy": {"weight": 0.10, "enabled": True},
    "sma_cross_strategy": {"weight": 0.10, "enabled": True},
    "pattern_recognition": {"weight": 0.10, "enabled": True},
    "sentiment_strategy": {"weight": 0.10, "enabled": True},
    "volume_analysis": {"weight": 0.10, "enabled": True}
}

# Economic Calendar Configuration
ECONOMIC_CALENDAR_UPDATE_INTERVAL = 3600  # 1 hour
ECONOMIC_EVENT_LOOKOUT_HOURS = 24  # Look for events 24 hours ahead

# Trading Parameters
SUPPORTED_ASSETS = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", 
    "MATIC", "DOT", "AVAX", "LINK", "UNI", "ATOM"
]
SUPPORTED_TIMEFRAMES = [1, 3, 5, 15]  # minutes

# Auto Asset Rotation Configuration
AUTO_ROTATION_ENABLED = False
AUTO_ROTATION_INTERVAL = 3600  # 1 hour

def get_api_keys() -> Dict[str, str]:
    """Get all API keys as a dictionary."""
    return {
        "alpha_vantage": ALPHA_VANTAGE_API_KEY,
        "coingecko": COINGECKO_API_KEY,
        "cryptocompare": CRYPTOCOMPARE_API_KEY,
        "coinmarketcap": COINMARKETCAP_API_KEY,
    }

def validate_config() -> bool:
    """Validate configuration settings."""
    required_vars = ["TELEGRAM_BOT_TOKEN"]
    
    # Check for missing required variables
    missing_vars = [var for var in required_vars if not globals().get(var)]
    
    if missing_vars:
        logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration."""
    return {
        "level": getattr(logging, LOG_LEVEL),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }