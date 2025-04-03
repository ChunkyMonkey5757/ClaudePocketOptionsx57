"""
Enhanced Bollinger Bands Strategy for PocketBotX57.
Implements advanced Bollinger Bands trading strategy with squeeze detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

from src.utils.logger import get_logger
from src.utils.error_handler import DataError, async_exception_handler

# Get logger
logger = get_logger("bollinger_strategy")

class BollingerStrategy:
    """
    Enhanced Bollinger Bands trading strategy.
    
    Features:
    - Dynamic standard deviation multiplier
    - Bollinger Band squeeze detection
    - Percent B calculation
    - Trend confirmation with price action
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Bollinger Bands strategy with parameters.
        
        Args:
            params: Dictionary with strategy parameters
        """
        # Default parameters
        self.name = "bollinger_strategy"
        self.enabled = True
        self.weight = 0.15
        
        # Bollinger specific parameters
        self.period = 20
        self.std_dev = 2.0
        self.use_dynamic_bands = True
        self.use_squeeze = True
        
        # Signal strength factors
        self.signal_strength = {
            "extreme_lower_band": 0.90,   # Price below lower band by >10%
            "lower_band": 0.75,           # Price at or below lower band
            "middle_band_up": 0.60,       # Price crosses middle band upward
            "middle_band_down": 0.60,     # Price crosses middle band downward
            "upper_band": 0.75,           # Price at or above upper band
            "extreme_upper_band": 0.90,   # Price above upper band by >10%
            "squeeze_breakout_up": 0.85,  # Breakout from squeeze upward
            "squeeze_breakout_down": 0.85 # Breakout from squeeze downward
        }
        
        # Override defaults with provided parameters
        if params:
            for key, value in params.items():
                if key == "signal_strength" and isinstance(value, dict):
                    self.signal_strength.update(value)
                elif hasattr(self, key):
                    setattr(self, key, value)
        
        logger.info(f"Bollinger Bands Strategy initialized with period={self.period}, "
                   f"std_dev={self.std_dev}")
    
    def validate_data(self, data: pd.DataFrame, required_columns: List[str] = None) -> bool:
        """
        Validate that required data is present.
        
        Args:
            data: Market data DataFrame
            required_columns: List of required column names
            
        Returns:
            Boolean indicating if data is valid
        """
        if required_columns is None:
            required_columns = ["close"]
            
        if data is None or data.empty:
            logger.warning("Empty dataset provided")
            return False
            
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
            
        # Need sufficient data points for Bollinger calculation
        if len(data) < self.period:
            logger.warning(f"Insufficient data points: {len(data)} < {self.period}")
            return False
            
        return True
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess market data for Bollinger Bands calculation.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Ensure index is datetime if not already
        if not isinstance(df.index, pd.DatetimeIndex) and 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        # Sort by index
        df = df.sort_index()
        
        # Fill any missing values
        df.fillna(method='ffill', inplace=True)
        
        return df
    
    def calculate_bollinger_bands(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands from market data.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary with Bollinger Bands components
        """
        if not self.validate_data(data):
            raise DataError("Invalid data for Bollinger Bands calculation")
            
        # Calculate middle band (SMA)
        middle_band = data['close'].rolling(window=self.period).mean()
        
        # Calculate standard deviation
        std = data['close'].rolling(window=self.period).std()
        
        # Calculate dynamic multiplier if enabled
        if self.use_dynamic_bands:
            # Adjust multiplier based on volatility (higher volatility = wider bands)
            volatility = (std / middle_band) * 100  # Normalized volatility as percentage
            
            # Calculate dynamic multiplier (ranges from 1.8 to 2.5 based on volatility)
            # Low volatility: tighter bands (1.8), High volatility: wider bands (2.5)
            multiplier = np.where(
                volatility.isna(),
                self.std_dev,  # Default when NA
                np.clip(1.8 + volatility * 0.05, 1.8, 2.5)  # Dynamic range
            )
        else:
            # Fixed multiplier
            multiplier = self.std_dev
        
        # Calculate upper and lower bands
        if isinstance(multiplier, pd.Series):
            upper_band = middle_band + (multiplier * std)
            lower_band = middle_band - (multiplier * std)
        else:
            upper_band = middle_band + (self.std_dev * std)
            lower_band = middle_band - (self.std_dev * std)
        
        # Calculate bandwidth (measure of band width, used for squeeze detection)
        bandwidth = (upper_band - lower_band) / middle_band * 100
        
        # Calculate %B (relative position within bands, 0-100%)
        percent_b = (data['close'] - lower_band) / (upper_band - lower_band) * 100
        percent_b = percent_b.clip(0, 100)  # Ensure it's between 0-100%
        
        return {
            "middle_band": middle_band,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "bandwidth": bandwidth,
            "percent_b": percent_b,
            "std_dev": std
        }
    
    def detect_band_touch(self, data: pd.DataFrame, bands: Dict[str, pd.Series]) -> Dict[str, bool]:
        """
        Detect when price touches or crosses Bollinger Bands.
        
        Args:
            data: Market data DataFrame
            bands: Dictionary with Bollinger Bands components
            
        Returns:
            Dictionary with band touch information
        """
        # Need at least 2 data points
        if len(data) < 2:
            return {
                "upper_band_touch": False,
                "lower_band_touch": False,
                "extreme_upper": False,
                "extreme_lower": False,
                "middle_cross_up": False,
                "middle_cross_down": False
            }
        
        # Get current and previous values
        curr_close = data['close'].iloc[-1]
        prev_close = data['close'].iloc[-2]
        
        curr_upper = bands["upper_band"].iloc[-1]
        curr_lower = bands["lower_band"].iloc[-1]
        curr_middle = bands["middle_band"].iloc[-1]
        
        prev_upper = bands["upper_band"].iloc[-2]
        prev_lower = bands["lower_band"].iloc[-2]
        prev_middle = bands["middle_band"].iloc[-2]
        
        # Detect band touches and crosses
        upper_band_touch = curr_close >= curr_upper
        lower_band_touch = curr_close <= curr_lower
        
        # Extreme touches (price significantly beyond bands)
        band_width = curr_upper - curr_lower
        extreme_upper = curr_close > curr_upper + (band_width * 0.1)  # 10% beyond upper band
        extreme_lower = curr_close < curr_lower - (band_width * 0.1)  # 10% beyond lower band
        
        # Middle band crosses
        middle_cross_up = prev_close < prev_middle and curr_close > curr_middle
        middle_cross_down = prev_close > prev_middle and curr_close < curr_middle
        
        return {
            "upper_band_touch": upper_band_touch,
            "lower_band_touch": lower_band_touch,
            "extreme_upper": extreme_upper,
            "extreme_lower": extreme_lower,
            "middle_cross_up": middle_cross_up,
            "middle_cross_down": middle_cross_down
        }
    
    def detect_squeeze(self, bands: Dict[str, pd.Series]) -> Dict[str, bool]:
        """
        Detect Bollinger Band squeeze and breakout.
        
        Args:
            bands: Dictionary with Bollinger Bands components
            
        Returns:
            Dictionary with squeeze information
        """
        # Need at least 10 data points for squeeze detection
        bandwidth = bands["bandwidth"]
        if len(bandwidth) < 10:
            return {"squeeze": False, "breakout_up": False, "breakout_down": False}
        
        # Calculate recent bandwidth statistics
        recent_bandwidth = bandwidth.iloc[-10:]
        min_bandwidth = recent_bandwidth.min()
        avg_bandwidth = recent_bandwidth.mean()
        
        # Current values
        curr_bandwidth = bandwidth.iloc[-1]
        prev_bandwidth = bandwidth.iloc[-2] if len(bandwidth) > 1 else curr_bandwidth
        
        # Detect squeeze (bandwidth narrowing significantly)
        # Squeeze is when bandwidth is in the lowest 20% of recent range
        squeeze_threshold = min_bandwidth + (avg_bandwidth - min_bandwidth) * 0.2
        squeeze = curr_bandwidth <= squeeze_threshold
        
        # Detect breakout from squeeze
        # Breakout is when bandwidth expands after being in a squeeze
        breakout = prev_bandwidth <= squeeze_threshold and curr_bandwidth > squeeze_threshold
        
        # Determine breakout direction using price action
        breakout_up = False
        breakout_down = False
        
        if breakout:
            # Use percent B to determine direction
            percent_b = bands["percent_b"]
            curr_percent_b = percent_b.iloc[-1]
            
            if curr_percent_b > 60:  # Upper half of bands
                breakout_up = True
            elif curr_percent_b < 40:  # Lower half of bands
                breakout_down = True
        
        return {
            "squeeze": squeeze,
            "breakout_up": breakout_up,
            "breakout_down": breakout_down
        }
    
    def determine_trend(self, data: pd.DataFrame, bands: Dict[str, pd.Series]) -> str:
        """
        Determine overall price trend using Bollinger Bands.
        
        Args:
            data: Market data DataFrame
            bands: Dictionary with Bollinger Bands components
            
        Returns:
            Trend direction: "bullish", "bearish", or "neutral"
        """
        try:
            # Use position relative to middle band for trend
            if len(data) < 5:
                return "neutral"
                
            # Get recent closes and middle band
            recent_close = data['close'].iloc[-5:]
            recent_middle = bands["middle_band"].iloc[-5:]
            
            # Count how many closes are above/below middle band
            above_count = sum(recent_close > recent_middle)
            below_count = sum(recent_close < recent_middle)
            
            # Strong trend if majority of points are on one side
            if above_count >= 4:  # 4 or 5 points above
                return "bullish"
            elif below_count >= 4:  # 4 or 5 points below
                return "bearish"
            
            # Check for rising/falling middle band
            if bands["middle_band"].iloc[-1] > bands["middle_band"].iloc[-5] * 1.005:  # 0.5% rise
                return "bullish"
            elif bands["middle_band"].iloc[-1] < bands["middle_band"].iloc[-5] * 0.995:  # 0.5% fall
                return "bearish"
            
            # Default to neutral
            return "neutral"
            
        except Exception as e:
            logger.warning(f"Error determining trend: {str(e)}")
            return "neutral"
    
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading signal based on Bollinger Bands strategy.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Signal dictionary with direction and confidence
        """
        try:
            if not self.validate_data(data):
                return {
                    "direction": "NEUTRAL",
                    "confidence": 0,
                    "details": {"error": "Invalid data"}
                }
                
            # Preprocess data
            df = self.preprocess_data(data)
            
            # Calculate Bollinger Bands
            bands = self.calculate_bollinger_bands(df)
            
            # Ensure we have valid band values
            if bands["middle_band"].isna().all():
                return {
                    "direction": "NEUTRAL",
                    "confidence": 0,
                    "details": {"error": "Invalid Bollinger Bands calculation"}
                }
                
            # Get current values
            current_close = df['close'].iloc[-1]
            current_upper = bands["upper_band"].iloc[-1]
            current_lower = bands["lower_band"].iloc[-1]
            current_middle = bands["middle_band"].iloc[-1]
            current_percent_b = bands["percent_b"].iloc[-1]
            
            # Detect band touches and crosses
            touches = self.detect_band_touch(df, bands)
            
            # Detect squeeze conditions
            squeeze_info = {}
            if self.use_squeeze:
                squeeze_info = self.detect_squeeze(bands)
            
            # Determine trend
            trend = self.determine_trend(df, bands)
            
            # Initialize signal
            signal = {
                "direction": "NEUTRAL",
                "confidence": 0,
                "details": {
                    "upper_band": current_upper,
                    "middle_band": current_middle,
                    "lower_band": current_lower,
                    "percent_b": current_percent_b,
                    "bandwidth": bands["bandwidth"].iloc[-1],
                    "trend": trend,
                    "upper_band_touch": touches.get("upper_band_touch", False),
                    "lower_band_touch": touches.get("lower_band_touch", False),
                    "middle_cross_up": touches.get("middle_cross_up", False),
                    "middle_cross_down": touches.get("middle_cross_down", False),
                    "squeeze": squeeze_info.get("squeeze", False),
                    "breakout_up": squeeze_info.get("breakout_up", False),
                    "breakout_down": squeeze_info.get("breakout_down", False)
                }
            }
            
            # Generate signal based on Bollinger Band conditions
            
            # Squeeze breakout signals (highest priority)
            if squeeze_info.get("breakout_up", False):
                signal["direction"] = "BUY"
                confidence = self.signal_strength["squeeze_breakout_up"]
                
                # Adjust confidence based on trend
                if trend == "bullish":
                    confidence += 0.10
                elif trend == "bearish":
                    confidence -= 0.15
                
                signal["confidence"] = min(confidence * 100, 95)  # Cap at 95%
                
            elif squeeze_info.get("breakout_down", False):
                signal["direction"] = "SELL"
                confidence = self.signal_strength["squeeze_breakout_down"]
                
                # Adjust confidence based on trend
                if trend == "bearish":
                    confidence += 0.10
                elif trend == "bullish":
                    confidence -= 0.15
                
                signal["confidence"] = min(confidence * 100, 95)  # Cap at 95%
                
            # Lower band touches/breaks
            elif touches.get("extreme_lower", False):
                signal["direction"] = "BUY"
                confidence = self.signal_strength["extreme_lower_band"]
                
                # Reduce confidence if overall trend is bearish
                if trend == "bearish":
                    confidence -= 0.15
                
                signal["confidence"] = min(confidence * 100, 95)  # Cap at 95%
                
            elif touches.get("lower_band_touch", False):
                signal["direction"] = "BUY"
                confidence = self.signal_strength["lower_band"]
                
                # Adjust confidence based on percent B
                # Lower percent B means deeper penetration of lower band
                if current_percent_b < 5:  # Deep penetration
                    confidence += 0.10
                
                # Reduce confidence if overall trend is bearish
                if trend == "bearish":
                    confidence -= 0.15
                
                signal["confidence"] = confidence * 100
                
            # Upper band touches/breaks
            elif touches.get("extreme_upper", False):
                signal["direction"] = "SELL"
                confidence = self.signal_strength["extreme_upper_band"]
                
                # Reduce confidence if overall trend is bullish
                if trend == "bullish":
                    confidence -= 0.15
                
                signal["confidence"] = min(confidence * 100, 95)  # Cap at 95%
                
            elif touches.get("upper_band_touch", False):
                signal["direction"] = "SELL"
                confidence = self.signal_strength["upper_band"]
                
                # Adjust confidence based on percent B
                # Higher percent B means deeper penetration of upper band
                if current_percent_b > 95:  # Deep penetration
                    confidence += 0.10
                
                # Reduce confidence if overall trend is bullish
                if trend == "bullish":
                    confidence -= 0.15
                
                signal["confidence"] = confidence * 100
                
            # Middle band crosses
            elif touches.get("middle_cross_up", False):
                signal["direction"] = "BUY"
                confidence = self.signal_strength["middle_band_up"]
                
                # Increase confidence if trend agrees
                if trend == "bullish":
                    confidence += 0.10
                
                signal["confidence"] = confidence * 100
                
            elif touches.get("middle_cross_down", False):
                signal["direction"] = "SELL"
                confidence = self.signal_strength["middle_band_down"]
                
                # Increase confidence if trend agrees
                if trend == "bearish":
                    confidence += 0.10
                
                signal["confidence"] = confidence * 100
                
            # Position-based signals (weakest)
            elif current_percent_b < 25 and trend != "bearish":
                # Low percent B without touching lower band yet
                signal["direction"] = "BUY"
                signal["confidence"] = 50  # Lower confidence
                
            elif current_percent_b > 75 and trend != "bullish":
                # High percent B without touching upper band yet
                signal["direction"] = "SELL"
                signal["confidence"] = 50  # Lower confidence
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating Bollinger Bands signal: {str(e)}")
            return {
                "direction": "NEUTRAL",
                "confidence": 0,
                "details": {"error": str(e)}
            }