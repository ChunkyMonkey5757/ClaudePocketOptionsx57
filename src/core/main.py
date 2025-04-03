
"""
Main Entry Point for PocketBotX57 AI Trading System.
Handles full system startup, strategy registration, scheduler loop, watchdog monitoring,
Telegram bot launch, and service initialization.
"""

import asyncio
import signal
import sys
from src.config import settings
from src.telegram_bot.bot import TelegramBot
from src.watchdog import Watchdog
from src.scheduler import TaskScheduler
from src.signal_engine.signal_engine import SignalEngine
from src.api_handler.api_manager import APIManager
from src.utils.logger import get_logger

logger = get_logger("main")

# Core services and containers
bot = TelegramBot()
scheduler = TaskScheduler()
watchdog = Watchdog(timeout_seconds=120)
api_manager = APIManager()
signal_engine = SignalEngine(api_manager=api_manager)

# Active user session map
user_sessions = {}

def graceful_shutdown(*args):
    """
    Handles graceful shutdown of async tasks and services.
    """
    logger.warning("Shutdown signal received. Cleaning up...")
    for task in asyncio.all_tasks():
        task.cancel()
    sys.exit(0)

async def signal_callback(symbol: str, user_id: int):
    """
    Async callback used by scheduler to scan an asset and send signals to Telegram.

    Args:
        symbol: Asset symbol (e.g., BTC, ETH)
        user_id: Telegram user ID
    """
    try:
        logger.info(f"[{user_id}] Scanning {symbol}")
        result = await signal_engine.generate_signal(symbol, interval="1m")

        if not result or result["direction"] == "NEUTRAL":
            logger.info(f"[{user_id}] No signal for {symbol}")
            return

        await bot.send_signal(
            user_id=user_id,
            symbol=symbol,
            signal=result
        )

        # Update watchdog heartbeat
        watchdog.beat("signal_loop")

    except Exception as e:
        logger.error(f"Signal callback error [{symbol}, {user_id}]: {str(e)}")

async def launch_bot():
    """
    Launches Telegram bot polling, registers async handlers and watchdogs.
    """
    logger.info("Launching PocketBotX57 System Core...")
    await bot.initialize()

    # Register default scanning session for testing (can be dynamic later)
    test_user_id = settings.ADMIN_USER_ID or 123456789
    asset_list = ["BTC", "ETH", "SOL"]

    logger.info(f"Starting scheduler loop for test user: {test_user_id}")
    scheduler.start_scanning(
        user_id=test_user_id,
        symbol_list=asset_list,
        interval_seconds=60,
        callback=signal_callback
    )

    # Register signal scanning with watchdog
    watchdog.register_task("signal_loop", lambda: scheduler.start_scanning(
        user_id=test_user_id,
        symbol_list=asset_list,
        interval_seconds=60,
        callback=signal_callback
    ))

    # Launch watchdog
    asyncio.create_task(watchdog.monitor(interval=15))

    # Start Telegram bot
    logger.info("Starting Telegram polling loop...")
    await bot.application.run_polling()

def setup_signals():
    """
    Attach system signal handlers for graceful shutdown.
    """
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

if __name__ == "__main__":
    setup_signals()
    try:
        asyncio.run(launch_bot())
    except Exception as e:
        logger.critical(f"Fatal PocketBotX57 Crash: {str(e)}", exc_info=True)
