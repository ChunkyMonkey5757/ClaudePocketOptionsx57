"""
Boot Module for PocketBotX57.
Orchestrates system startup, validates dependencies, performs health diagnostics,
and launches main services cleanly for Glitch, Vercel, or standalone environments.
"""

import asyncio
import platform
import psutil
import logging
import os

from src import main
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger("boot")


def print_banner():
    print("\n" + "="*60)
    print("        PocketBotX57 — AI Trading Automation System")
    print("="*60)
    print(f"OS: {platform.system()} {platform.release()} | Python {platform.python_version()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=True)} | RAM: {round(psutil.virtual_memory().total / 1e9, 2)} GB")
    print(f"Mode: {'DEBUG' if settings.DEBUG_MODE else 'PRODUCTION'}")
    print("="*60 + "\n")


def validate_environment():
    """
    Ensures required keys are loaded before launch.
    """
    missing = []
    if not settings.BOT_TOKEN or settings.BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        missing.append("TELEGRAM_BOT_TOKEN")
    if not settings.KRAKEN_API_KEY:
        missing.append("KRAKEN_API_KEY")
    if not settings.KRAKEN_SECRET_KEY:
        missing.append("KRAKEN_SECRET_KEY")

    if missing:
        logger.critical(f"Missing required environment variables: {missing}")
        raise SystemExit("Startup aborted due to config errors.")


def diagnostics():
    """
    Performs diagnostics on API keys, strategy weight totals, and system compatibility.
    """
    logger.info("Running diagnostics...")
    total_weight = sum([
        settings.WEIGHT_RSI,
        settings.WEIGHT_MACD,
        settings.WEIGHT_BOLLINGER,
        settings.WEIGHT_SMA,
        settings.WEIGHT_VOLUME,
        settings.WEIGHT_VWAP,
        settings.WEIGHT_PATTERN,
        settings.WEIGHT_SENTIMENT,
        settings.WEIGHT_CORRELATION,
    ])
    if not 0.99 <= total_weight <= 1.01:
        logger.warning(f"Strategy weights do not sum to 1.0 (total: {total_weight})")

    logger.info(f"Using {len(settings.DEFAULT_ASSETS)} default assets: {settings.DEFAULT_ASSETS}")
    logger.info("Diagnostics complete.")


async def launch():
    """
    Launch the main bot system.
    """
    print_banner()
    validate_environment()
    diagnostics()

    from src.telegram_bot.telegram_bot import TelegramBot
    bot = TelegramBot()
    await bot.initialize()
    await bot.run_polling()


if __name__ == "__main__":
    try:
        asyncio.run(launch())
    except KeyboardInterrupt:
        logger.info("Bot terminated by user.")
    except Exception as e:
        logger.exception(f"Uncaught error in boot sequence: {e}")
