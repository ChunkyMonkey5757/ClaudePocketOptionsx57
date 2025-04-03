
"""
AI Learning Tracker for PocketBotX57.
Logs trade results, analyzes strategy performance, and evolves weights over time.
"""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger("ai_learning_tracker")

TRACKER_PATH = "storage/trade_log.json"

class AILearningTracker:
    """
    Tracks trade outcomes and analyzes performance by strategy and confidence score.
    """

    def __init__(self):
        self.log_file = TRACKER_PATH
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self._load()

    def _load(self):
        try:
            with open(self.log_file, "r") as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            self.data = []

    def _save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def record_result(self, symbol: str, direction: str, confidence: float, result: str, strategies: List[str]):
        """
        Log the outcome of a signal/trade.

        Args:
            symbol: Asset traded
            direction: BUY or SELL
            confidence: Signal confidence
            result: "won" or "lost"
            strategies: list of strategy names used
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "result": result,
            "strategies": strategies
        }
        self.data.append(entry)
        self._save()
        logger.info(f"Recorded result: {entry}")

    def summarize(self) -> Dict[str, Dict[str, float]]:
        """
        Summarizes strategy accuracy from logged results.

        Returns:
            Summary dict with win/loss rate per strategy
        """
        stats = defaultdict(lambda: {"won": 0, "lost": 0, "win_rate": 0})

        for entry in self.data:
            for strat in entry["strategies"]:
                if entry["result"] == "won":
                    stats[strat]["won"] += 1
                elif entry["result"] == "lost":
                    stats[strat]["lost"] += 1

        for strat in stats:
            total = stats[strat]["won"] + stats[strat]["lost"]
            if total > 0:
                stats[strat]["win_rate"] = round((stats[strat]["won"] / total) * 100, 2)

        return dict(stats)

    def get_recent(self, count: int = 10) -> List[Dict]:
        """
        Returns most recent trade logs.

        Args:
            count: Number of entries

        Returns:
            List of recent entries
        """
        return self.data[-count:]
