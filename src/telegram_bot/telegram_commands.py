
"""
Telegram Command Handlers for PocketBotX57.
Processes user commands like /won, /lost, /summary, /weights, /history.
"""

from telegram import Update
from telegram.ext import ContextTypes
from src.ai_learning_tracker import AILearningTracker
from src.signal_journal import SignalJournal
from src.stats_engine import StatsEngine
from src.adaptive_weights import AdaptiveWeightEngine
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger("telegram_commands")

# Global instances
tracker = AILearningTracker()
journal = SignalJournal()
stats = StatsEngine()
weight_engine = AdaptiveWeightEngine()

# Used to temporarily store last signal for logging
user_last_signal = {}

async def won(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    last = user_last_signal.get(user_id)
    if not last:
        await update.message.reply_text("No signal found to log as a win.")
        return
    tracker.record_result(**last, result="won")
    await update.message.reply_text("Result recorded as a WIN.")

async def lost(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    last = user_last_signal.get(user_id)
    if not last:
        await update.message.reply_text("No signal found to log as a loss.")
        return
    tracker.record_result(**last, result="lost")
    await update.message.reply_text("Result recorded as a LOSS.")

async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    report = stats.get_report()
    await update.message.reply_text(f"```\n{report}\n```", parse_mode="Markdown")

async def weights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    evolved = weight_engine.evolve_weights()
    lines = [f"{k}: {v}" for k, v in evolved.items()]
    await update.message.reply_text("Updated Strategy Weights:\n" + "\n".join(lines))

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    recent = journal.get_recent(5)
    if not recent:
        await update.message.reply_text("No recent signals recorded.")
        return
    msgs = []
    for r in recent:
        msg = f"{r['timestamp']}\n{r['symbol']} | {r['direction']} | {r['confidence']}% for {r['duration']}m"
        msgs.append(msg)
    await update.message.reply_text("\n\n".join(msgs))

# Helper: log signal during live signal generation
def store_last_signal(user_id: int, symbol: str, direction: str, confidence: float, strategies: list):
    user_last_signal[user_id] = {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "strategies": strategies
    }
