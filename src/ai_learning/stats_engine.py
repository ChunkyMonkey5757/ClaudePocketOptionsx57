
"""
Statistics Engine for PocketBotX57.
Tracks session performance, win/loss ratio, average confidence, ROI estimate, etc.
"""

from typing import List, Dict
from collections import defaultdict
from src.ai_learning_tracker import AILearningTracker
from src.utils.logger import get_logger

logger = get_logger("stats_engine")

class StatsEngine:
    """
    Aggregates and reports signal/trade performance metrics.
    """

    def __init__(self):
        self.tracker = AILearningTracker()

    def get_stats(self) -> Dict[str, float]:
        """
        Calculates global win/loss performance across all strategies.

        Returns:
            Dict of metrics
        """
        data = self.tracker.data
        total = len(data)
        if total == 0:
            return {"total_trades": 0, "win_rate": 0, "avg_confidence": 0}

        won = sum(1 for x in data if x["result"] == "won")
        avg_conf = sum(x["confidence"] for x in data) / total
        win_rate = (won / total) * 100

        return {
            "total_trades": total,
            "wins": won,
            "losses": total - won,
            "win_rate": round(win_rate, 2),
            "avg_confidence": round(avg_conf, 2)
        }

    def get_strategy_breakdown(self) -> Dict[str, Dict[str, float]]:
        """
        Breaks down win/loss counts per strategy.

        Returns:
            Dict of stats by strategy
        """
        return self.tracker.summarize()

    def get_report(self) -> str:
        """
        Builds a printable report of performance metrics.

        Returns:
            Markdown-style string
        """
        global_stats = self.get_stats()
        strategy_stats = self.get_strategy_breakdown()

        lines = [
            "===== PocketBotX57 Performance Summary =====",
            f"Total Trades: {global_stats['total_trades']}",
            f"Win Rate: {global_stats['win_rate']}%",
            f"Avg Confidence: {global_stats['avg_confidence']}%",
            ""
        ]

        lines.append("Per-Strategy Breakdown:")
        for strat, stats in strategy_stats.items():
            lines.append(f"- {strat}: {stats['won']}W / {stats['lost']}L (Win Rate: {stats['win_rate']}%)")

        return "\n".join(lines)
