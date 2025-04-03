"""
MACD Strategy module for PocketBotX57.
Implements the Moving Average Convergence Divergence trading strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union

from src.indicators.base_indicator import BaseIndicator
from src.utils.error_handler import DataError

class MACDStrategy(BaseIndicator):
    """
    Moving Average Convergence Divergence (MACD) trading strategy.
    
    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price. The MACD is calculated by
    subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA.
    A 9-day EMA of the MACD, called the "signal line", is then plotted on top of the MACD,
    functioning as a trigger for buy and sell signals.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the MACD strategy.
        
        Args:
            params (Dict[str, Any], optional): Strategy parameters. Defaults to None.
        """
        default_params = {
            "weight": 0.20,
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "signal_strength": {
                "strong_bullish_cross": 0.90,  # MACD crosses above signal from below with separation
                "bullish_cross": 0.75,        # MACD crosses above signal
                "bullish_momentum": 0.60,     # MACD and signal both rising
                "bearish_momentum": 0.60,     # MACD and signal both falling
                "bearish_cross": 0.75,        # MACD crosses below signal
                "strong_bearish_cross": 0.90   # MACD crosses below signal from above with separation
            }
        }
        
        # Merge default params with provided params
        if params:
            for key, value in params.items():
                if key == "signal_strength" and isinstance(value, dict):
                    default_params["signal_strength"].update(value)
                else:
                    default_params[key] = value
        
        super().__init__("macd_strategy", default_params)
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate MACD values from market data.
        
        Args:
            data (pd.DataFrame): Market data with OHLC values
            
        Returns:
            Dict[str, Any]: MACD values and analysis
            
        Raises:
            DataError: If calculation fails
        """
        try:
            # Validate data
            self.validate_data(data, ["close"])
            
            # Preprocess data
            df = self.preprocess_data(data)
            
            # Get MACD parameters
            fast_period = self.params["fast_period"]
            slow_period = self.params["slow_period"]
            signal_period = self.params["signal_period"]
            
            # Calculate MACD
            # Fast EMA
            ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()
            
            # Slow EMA
            ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()
            
            # MACD Line
            macd_line = ema_fast - ema_slow
            
            # Signal Line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # MACD Histogram
            histogram = macd_line - signal_line
            
            # Get the latest values
            latest_macd = macd_line.iloc[-1]
            latest_signal = signal_line.iloc[-1]
            latest_histogram = histogram.iloc[-1]
            
            # Previous values
            prev_macd = macd_line.iloc[-2]
            prev_signal = signal_line.iloc[-2]
            prev_histogram = histogram.iloc[-2]
            
            # MACD trend
            macd_trend = "increasing" if latest_macd > prev_macd else "decreasing"
            signal_trend = "increasing" if latest_signal > prev_signal else "decreasing"
            histogram_trend = "increasing" if latest_histogram > prev_histogram else "decreasing"
            
            # Crossover detection
            bullish_cross = prev_macd < prev_signal and latest_macd > latest_signal
            bearish_cross = prev_macd > prev_signal and latest_macd < latest_signal
            
            # Momentum measurement (histogram direction change)
            bullish_momentum = prev_histogram < 0 and latest_histogram > 0
            bearish_momentum = prev_histogram > 0 and latest_histogram < 0
            
            # Divergence detection (price vs. MACD)
            price_direction = 1 if df["close"].iloc[-1] > df["close"].iloc[-5] else -1
            macd_direction = 1 if latest_macd > macd_line.iloc[-5] else -1
            
            divergence = price_direction != macd_direction
            bullish_divergence = divergence and price_direction < 0 and macd_direction > 0
            bearish_divergence = divergence and price_direction > 0 and macd_direction < 0
            
            # Strength of signal measurement
            macd_above_signal = latest_macd > latest_signal
            histogram_strength = abs(latest_histogram) / (abs(latest_macd) + 1e-10)  # Avoid division by zero
            
            return {
                "macd_line": macd_line.to_list(),
                "signal_line": signal_line.to_list(),
                "histogram": histogram.to_list(),
                "latest_macd": latest_macd,
                "latest_signal": latest_signal,
                "latest_histogram": latest_histogram,
                "macd_trend": macd_trend,
                "signal_trend": signal_trend,
                "histogram_trend": histogram_trend,
                "bullish_cross": bullish_cross,
                "bearish_cross": bearish_cross,
                "bullish_momentum": bullish_momentum,
                "bearish_momentum": bearish_momentum,
                "macd_above_signal": macd_above_signal,
                "histogram_strength": histogram_strength,
                "divergence": divergence,
                "bullish_divergence": bullish_divergence,
                "bearish_divergence": bearish_divergence
            }
            
        except Exception as e:
            raise DataError(f"MACD calculation failed: {str(e)}")
    
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get trading signal based on MACD values.
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            Dict[str, Any]: Trading signal with direction and confidence
            
        Raises:
            DataError: If signal generation fails
        """
        try:
            # Calculate MACD
            macd_data = self.calculate(data)
            
            # Initialize signal
            signal = {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "details": macd_data
            }
            
            # Strong bullish cross (with good separation)
            if macd_data["bullish_cross"] and macd_data["histogram_strength"] > 0.1:
                signal["direction"] = "BUY"
                confidence = self.params["signal_strength"]["strong_bullish_cross"]
                
                # Increase confidence on bullish divergence
                if macd_data["bullish_divergence"]:
                    confidence += 0.10
                
                signal["confidence"] = min(confidence * 100, 95)  # Cap at 95%
                
            # Normal bullish cross
            elif macd_data["bullish_cross"]:
                signal["direction"] = "BUY"
                signal["confidence"] = self.params["signal_strength"]["bullish_cross"] * 100
                
            # Bullish momentum (MACD above signal and both rising)
            elif (macd_data["macd_above_signal"] and 
                  macd_data["macd_trend"] == "increasing" and 
                  macd_data["signal_trend"] == "increasing"):
                signal["direction"] = "BUY"
                signal["confidence"] = self.params["signal_strength"]["bullish_momentum"] * 100
                
            # Strong bearish cross (with good separation)
            elif macd_data["bearish_cross"] and macd_data["histogram_strength"] > 0.1:
                signal["direction"] = "SELL"
                confidence = self.params["signal_strength"]["strong_bearish_cross"]
                
                # Increase confidence on bearish divergence
                if macd_data["bearish_divergence"]:
                    confidence += 0.10
                
                signal["confidence"] = min(confidence * 100, 95)  # Cap at 95%
                
            # Normal bearish cross
            elif macd_data["bearish_cross"]:
                signal["direction"] = "SELL"
                signal["confidence"] = self.params["signal_strength"]["bearish_cross"] * 100
                
            # Bearish momentum (MACD below signal and both falling)
            elif (not macd_data["macd_above_signal"] and 
                  macd_data["macd_trend"] == "decreasing" and 
                  macd_data["signal_trend"] == "decreasing"):
                signal["direction"] = "SELL"
                signal["confidence"] = self.params["signal_strength"]["bearish_momentum"] * 100
                
            # Histogram direction change (early signal)
            elif macd_data["bullish_momentum"]:
                signal["direction"] = "BUY"
                signal["confidence"] = 50  # Lower confidence for early signal
                
            elif macd_data["bearish_momentum"]:
                signal["direction"] = "SELL"
                signal["confidence"] = 50  # Lower confidence for early signal
                
            else:
                # No clear signal
                signal["direction"] = "NEUTRAL"
                signal["confidence"] = 0.0
            
            return signal
            
        except Exception as e:
            raise DataError(f"MACD signal generation failed: {str(e)}")