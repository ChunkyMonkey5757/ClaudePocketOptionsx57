"""
Enhanced RSI Strategy for PocketBotX57.
Implements advanced Relative Strength Index trading strategy with adaptive thresholds.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import asyncio

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
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dictionary with RSI values and analysis
        """
        try:
            # Validate data
            self._validate_data(df, required_columns=["close"])
            
            # Preprocess data
            df = self._preprocess_data(df)
            
            # Calculate RSI
            rsi = self._calculate_rsi(df)
            
            # Detect divergences
            divergences = self._detect_divergence(df, rsi) if self.use_divergence else {}
            
            # Calculate dynamic thresholds based on market conditions
            if self.dynamic_thresholds:
                adjusted_oversold, adjusted_overbought = self._adjust_thresholds(df)
            else:
                adjusted_oversold, adjusted_overbought = self.oversold, self.overbought
            
            # Get current RSI value
            current_rsi = rsi.iloc[-1]
            
            # Determine trend using moving averages
            trend = "neutral"
            if len(df) >= 20:
                sma20 = df['close'].rolling(window=20).mean()
                if df['close'].iloc[-1] > sma20.iloc[-1]:
                    trend = "bullish"
                elif df['close'].iloc[-1] < sma20.iloc[-1]:
                    trend = "bearish"
            
            # Multiple timeframe confirmation
            mtf_confirmation = self._check_multiple_timeframes(df)
            
            # Compile results
            result = {
                "rsi": rsi.tolist(),
                "current_rsi": current_rsi,
                "oversold": adjusted_oversold,
                "overbought": adjusted_overbought,
                "is_oversold": current_rsi < adjusted_oversold,
                "is_overbought": current_rsi > adjusted_overbought,
                "trend": trend,
                "mtf_confirmation": mtf_confirmation
            }
            
            # Add divergence data if available
            if divergences:
                result.update(divergences)
            
            return result
            
        except Exception as e:
            logger.error("RSI calculation error: %s", str(e))
            raise DataError(f"RSI calculation failed: {str(e)}")
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI from price data.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Series of RSI values
        """
        # Calculate price changes
        delta = df['close'].diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate average gains and losses
        avg_gain = gain.ewm(com=self.period-1, min_periods=self.period).mean()
        avg_loss = loss.ewm(com=self.period-1, min_periods=self.period).mean()
        
        # Calculate RS value
        rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _detect_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> Dict[str, bool]:
        """
        Detect price/RSI divergences.
        
        Args:
            df: DataFrame with OHLC data
            rsi: Series of RSI values
            
        Returns:
            Dictionary with divergence information
        """
        # Need at least 10 periods to detect divergence
        if len(df) < 10 or len(rsi) < 10:
            return {"bullish_divergence": False, "bearish_divergence": False}
            
        # Get recent prices and RSI values (last 10 periods)
        recent_close = df['close'].iloc[-10:].values
        recent_rsi = rsi.iloc[-10:].values
        
        # Find local extremes
        price_min_idx = np.argmin(recent_close)
        price_max_idx = np.argmax(recent_close)
        rsi_min_idx = np.argmin(recent_rsi)
        rsi_max_idx = np.argmax(recent_rsi)
        
        # Check for bullish divergence (price making lower lows but RSI making higher lows)
        bullish_divergence = False
        if price_min_idx >= 7 and rsi_min_idx <= 3:  # Recent price low but earlier RSI low
            price_trend = recent_close[-1] / recent_close[0]
            rsi_trend = recent_rsi[-1] / recent_rsi[0]
            bullish_divergence = price_trend < 1.0 and rsi_trend > 1.0
        
        # Check for bearish divergence (price making higher highs but RSI making lower highs)
        bearish_divergence = False
        if price_max_idx >= 7 and rsi_max_idx <= 3:  # Recent price high but earlier RSI high
            price_trend = recent_close[-1] / recent_close[0]
            rsi_trend = recent_rsi[-1] / recent_rsi[0]
            bearish_divergence = price_trend > 1.0 and rsi_trend < 1.0
        
        return {
            "bullish_divergence": bullish_divergence,
            "bearish_divergence": bearish_divergence
        }
    
    def _adjust_thresholds(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Dynamically adjust RSI thresholds based on market volatility.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Tuple of (adjusted_oversold, adjusted_overbought) thresholds
        """
        try:
            # Calculate volatility using ATR
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            
            # Normalize as percentage of price
            current_price = df['close'].iloc[-1]
            volatility = (atr / current_price) * 100
            
            # Adjust thresholds based on volatility
            # Higher volatility = wider thresholds
            # Lower volatility = tighter thresholds
            volatility_factor = min(1.5, max(0.5, volatility / 1.0))
            
            # Calculate adjusted thresholds
            adjusted_oversold = max(20, min(40, self.oversold - (volatility_factor - 1) * 10))
            adjusted_overbought = min(80, max(60, self.overbought + (volatility_factor - 1) * 10))
            
            return adjusted_oversold, adjusted_overbought
            
        except Exception as e:
            logger.warning("Error adjusting thresholds: %s", str(e))
            return self.oversold, self.overbought
    
    def _check_multiple_timeframes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check RSI across multiple timeframes for confirmation.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dictionary with multiple timeframe confirmation data
        """
        result = {
            "aligned": False,
            "direction": "NEUTRAL"
        }
        
        try:
            # Only proceed if we have enough data
            if len(df) < 120:
                return result
            
            # Calculate RSI for different timeframes
            # Original timeframe
            rsi_orig = self._calculate_rsi(df)
            
            # 3x timeframe (resample)
            df_3x = df.copy()
            df_3x = df_3x.iloc[::3]  # Sample every 3rd row
            rsi_3x = self._calculate_rsi(df_3x)
            
            # 5x timeframe (resample)
            df_5x = df.copy()
            df_5x = df_5x.iloc[::5]  # Sample every 5th row
            rsi_5x = self._calculate_rsi(df_5x)
            
            # Get current values
            current_rsi = rsi_orig.iloc[-1]
            current_rsi_3x = rsi_3x.iloc[-1]
            current_rsi_5x = rsi_5x.iloc[-1]
            
            # Check alignment
            bullish_aligned = (
                current_rsi < self.oversold and 
                current_rsi_3x < self.oversold and 
                current_rsi_5x < self.oversold
            )
            
            bearish_aligned = (
                current_rsi > self.overbought and 
                current_rsi_3x > self.overbought and 
                current_rsi_5x > self.overbought
            )
            
            if bullish_aligned:
                result["aligned"] = True
                result["direction"] = "BUY"
            elif bearish_aligned:
                result["aligned"] = True
                result["direction"] = "SELL"
            
            # Add additional data
            result["timeframes"] = {
                "original": current_rsi,
                "3x": current_rsi_3x,
                "5x": current_rsi_5x
            }
            
            return result
            
        except Exception as e:
            logger.warning("Error in multiple timeframe analysis: %s", str(e))
            return result
    
    def _validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that required data is present.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if valid, raises DataError otherwise
        """
        if df is None or df.empty:
            raise DataError("Empty dataset provided")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataError(f"Missing required columns: {missing_columns}")
        
        if len(df) < self.period:
            raise DataError(f"Insufficient data: {len(df)} rows, need at least {self.period}")
        
        return True
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for calculation.
        
        Args:
            df: DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Ensure index is datetime if not already
        if not isinstance(df.index, pd.DatetimeIndex) and 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        # Sort by index
        df = df.sort_index()
        
        # Fill any missing values
        df.fillna(method='ffill', inplace=True)
        
        return df
    
    async def get_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal based on RSI analysis.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Signal dictionary with direction and confidence
        """
        try:
            # Calculate RSI and related metrics
            rsi_data = await self.calculate(df)
            
            # Initialize signal
            signal = {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "indicators": {
                    "rsi": rsi_data["current_rsi"]
                }
            }
            
            current_rsi = rsi_data["current_rsi"]
            oversold = rsi_data["oversold"]
            overbought = rsi_data["overbought"]
            trend = rsi_data["trend"]
            
            # Generate signal based on RSI and other factors
            if current_rsi < oversold:
                # Oversold - potential BUY signal
                signal["direction"] = "BUY"
                
                # Determine confidence level
                if current_rsi < 20:
                    confidence = self.params["signal_strength"]["extreme_oversold"]
                else:
                    confidence = self.params["signal_strength"]["oversold"]
                
                # Increase confidence if bullish divergence is detected
                if rsi_data.get("bullish_divergence", False):
                    confidence = min(0.95, confidence + 0.15)
                
                # Increase confidence if confirmed across timeframes
                if rsi_data.get("mtf_confirmation", {}).get("aligned", False) and rsi_data.get("mtf_confirmation", {}).get("direction") == "BUY":
                    confidence = min(0.95, confidence + 0.10)
                
                # Reduce confidence if bearish trend
                if trend == "bearish":
                    confidence -= 0.1
                
                signal["confidence"] = min(confidence * 100, 95)  # Cap at 95%
                
            elif current_rsi > overbought:
                # Overbought - potential SELL signal
                signal["direction"] = "SELL"
                
                # Determine confidence level
                if current_rsi > 80:
                    confidence = self.params["signal_strength"]["extreme_overbought"]
                else:
                    confidence = self.params["signal_strength"]["overbought"]
                
                # Increase confidence if bearish divergence is detected
                if rsi_data.get("bearish_divergence", False):
                    confidence = min(0.95, confidence + 0.15)
                
                # Increase confidence if confirmed across timeframes
                if rsi_data.get("mtf_confirmation", {}).get("aligned", False) and rsi_data.get("mtf_confirmation", {}).get("direction") == "SELL":
                    confidence = min(0.95, confidence + 0.10)
                
                # Reduce confidence if bullish trend
                if trend == "bullish":
                    confidence -= 0.1
                
                signal["confidence"] = min(confidence * 100, 95)  # Cap at 95%
                
            elif current_rsi < 40:
                # Approaching oversold - weak BUY signal
                signal["direction"] = "BUY"
                confidence = self.params["signal_strength"]["approaching_oversold"]
                
                # Stronger signal if in bullish trend
                if trend == "bullish":
                    confidence += 0.1
                
                signal["confidence"] = confidence * 100
                
            elif current_rsi > 60:
                # Approaching overbought - weak SELL signal
                signal["direction"] = "SELL"
                confidence = self.params["signal_strength"]["approaching_overbought"]
                
                # Stronger signal if in bearish trend
                if trend == "bearish":
                    confidence += 0.1
                
                signal["confidence"] = confidence * 100
            
            # Add additional indicator data
            signal["indicators"].update({
                "trend": trend,
                "overbought": overbought,
                "oversold": oversold,
                "bullish_divergence": rsi_data.get("bullish_divergence", False),
                "bearish_divergence": rsi_data.get("bearish_divergence", False),
                "mtf_confirmation": rsi_data.get("mtf_confirmation", {}).get("aligned", False)
            })
            
            return signal
            
        except Exception as e:
            logger.error("RSI signal generation error: %s", str(e))
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e)
            }