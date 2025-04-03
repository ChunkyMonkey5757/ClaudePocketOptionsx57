
"""
Adaptive Weight Engine for PocketBotX57.
Dynamically adjusts strategy weights based on historical performance (AI Learning Tracker).
"""

from src.ai_learning_tracker import AILearningTracker
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger("adaptive_weights")

class AdaptiveWeightEngine:
    def __init__(self):
        self.tracker = AILearningTracker()
        self.default_weights = settings.STRATEGY_WEIGHTS.copy()

    def evolve_weights(self) -> dict:
        """
        Adjusts strategy weights based on win rates.

        Returns:
            Updated weight dict
        """
        stats = self.tracker.summarize()
        total = 0
        new_weights = {}

        for strat, base_weight in self.default_weights.items():
            win_rate = stats.get(strat, {}).get("win_rate", 50)
            modifier = win_rate / 100
            new_weight = base_weight * modifier
            new_weights[strat] = new_weight
            total += new_weight

        # Normalize
        for strat in new_weights:
            new_weights[strat] = round(new_weights[strat] / total, 3) if total > 0 else self.default_weights[strat]

        logger.info(f"Updated strategy weights: {new_weights}")
        return new_weights
