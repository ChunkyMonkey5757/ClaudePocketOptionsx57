"""
Enhanced RSI Strategy for PocketBotX57.
Implements advanced Relative Strength Index trading strategy with adaptive thresholds.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

from src.indicators.indicator_base import IndicatorBase
from src.utils.logger import get_logger
from src.utils.error_handler import DataError

logger = get_logger(__name__)

class RSIStrategy(IndicatorBase):
    """
    Enhanced Relative Strength Index (RSI) trading strategy.
    
    Features:
    - Dynamic overbought/oversold thresholds based on volatility
    - RSI divergence detection
    - Trend-adjusted signal generation
    - Multiple timeframe confirmation
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the RSI strategy with parameters."""
        # Default parameters
        default_params = {
            "name": "rsi",
            "enabled": True,
            "weight": 0.15,
            "period": 14,
            "overbought": 70,
            "oversold": 30,
            "dynamic_thresholds": True,
            "use_divergence": True,
            "signal_strength": {
                "extreme_oversold": 0.90,  # RSI < 20
                "oversold": 0.75,          # RSI < 30
                "approaching_oversold": 0.60,  # RSI < 40
                "neutral_low": 0.40,       # RSI 40-50
                "neutral_high": 0.40,      # RSI 50-60
                "approaching_overbought": 0.60,  # RSI > 60
                "overbought": 0.75,        # RSI > 70
                "extreme_overbought": 0.90  # RSI > 80
            }
        }
        
        # Merge with provided parameters
        merged_params = default_params.copy()
        if params:
            for key, value in params.items():
                if key == "signal_strength" and isinstance(value, dict):
                    merged_params["signal_strength"].update(value)
                else:
                    merged_params[key] = value
        
        super().__init__(merged_params)
        
        # Store specific RSI parameters for easy access
        self.period = self.params["period"]
        self.overbought = self.params["overbought"]
        self.oversold = self.params["oversold"]
        self.dynamic_thresholds = self.params["dynamic_thresholds"]
        self.use_divergence = self.params["use_divergence"]
        
        logger.info(
            "RSI Strategy initialized (period=%d, overbought=%d, oversold=%d)",
            self.period, self.overbought, self.oversold
        )
    
    async def calculate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate RSI values and related metrics.
        
        Args