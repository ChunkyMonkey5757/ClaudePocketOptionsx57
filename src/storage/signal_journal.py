
"""
Signal Journal for PocketBotX57.
Maintains a readable log of signals, used by Telegram /history command.
"""

import json
import os
from datetime import datetime
from typing import List, Dict

JOURNAL_PATH = "storage/signal_journal.json"

class SignalJournal:
    def __init__(self):
        os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
        self._load()

    def _load(self):
        try:
            with open(JOURNAL_PATH, "r") as f:
                self.journal = json.load(f)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            self.journal = []

    def _save(self):
        with open(JOURNAL_PATH, "w") as f:
            json.dump(self.journal, f, indent=2)

    def log_signal(self, symbol: str, direction: str, confidence: float, duration: int, commentary: str = ""):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "duration": duration,
            "commentary": commentary
        }
        self.journal.append(entry)
        self._save()

    def get_recent(self, count: int = 10) -> List[Dict]:
        return self.journal[-count:]
