
"""
Telegram Bot Controller for PocketBotX57.
Handles command registration and user interaction routing.
"""

import os
from telegram.ext import Application, CommandHandler
from src.telegram_commands import won, lost, summary, weights, history
from src.utils.logger import get_logger
from src.config import settings

logger = get_logger("telegram_bot")

class TelegramBot:
    def __init__(self):
        self.token = settings.BOT_TOKEN
        self.application = None

    async def initialize(self):
        """
        Initializes bot and registers all command handlers.
        """
        logger.info("Initializing Telegram Bot...")
        self.application = Application.builder().token(self.token).build()

        self._register_commands()

    def _register_commands(self):
        """
        Add bot command handlers.
        """
        self.application.add_handler(CommandHandler("won", won))
        self.application.add_handler(CommandHandler("lost", lost))
        self.application.add_handler(CommandHandler("summary", summary))
        self.application.add_handler(CommandHandler("weights", weights))
        self.application.add_handler(CommandHandler("history", history))
        logger.info("Telegram commands registered.")

    async def send_signal(self, user_id: int, symbol: str, signal: dict):
        """
        Sends a formatted trade signal to a user.

        Args:
            user_id: Telegram ID
            symbol: Asset symbol
            signal: Dict with direction, confidence, duration, commentary
        """
        direction = signal.get("direction", "NEUTRAL")
        confidence = signal.get("confidence", 0)
        duration = signal.get("duration", 5)
        commentary = signal.get("commentary", "")

        text = f"Signal: {symbol}\nDirection: {direction}\nConfidence: {confidence}%\nDuration: {duration} min"
        if commentary:
            text += f"\nNote: {commentary}"

        await self.application.bot.send_message(chat_id=user_id, text=text)

        # Log to journal
        from src.signal_journal import SignalJournal
        from src.telegram_commands import store_last_signal
        journal = SignalJournal()
        journal.log_signal(symbol, direction, confidence, duration, commentary)
        store_last_signal(user_id, symbol, direction, confidence, signal.get("strategies", []))

    async def run_polling(self):
        await self.application.run_polling()
