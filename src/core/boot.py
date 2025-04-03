
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
    print("Mode: DEBUG" if settings.DEBUG_MODE else "Mode: PRODUCTION")
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
    logger.info("Performing diagnostics...")
    total_weight = sum(settings.STRATEGY_WEIGHTS.values())
    if abs(total_weight - 1.0) > 0.2:
        logger.warning(f"Total strategy weights = {round(total_weight, 2)} (expected ~1.0)")

    logger.info(f"Strategies Loaded: {list(settings.STRATEGY_WEIGHTS.keys())}")
    logger.info(f"Default Assets: {settings.DEFAULT_ASSETS}")
    logger.info(f"Watchdog Timeout: {settings.WATCHDOG_TIMEOUT_SECONDS}s")
    logger.info(f"Cooldown Timer: {settings.SIGNAL_COOLDOWN_SECONDS}s")

def start_launcher():
    """
    Entrypoint for external cloud platforms or CLI.
    """
    try:
        print_banner()
        validate_environment()
        diagnostics()
        asyncio.run(main.launch_bot())
    except KeyboardInterrupt:
        logger.warning("Manual interrupt received. Shutting down.")
    except Exception as e:
        logger.critical(f"Unhandled boot error: {str(e)}", exc_info=True)
        raise SystemExit("Boot failure.")

if __name__ == "__main__":
    start_launcher()
