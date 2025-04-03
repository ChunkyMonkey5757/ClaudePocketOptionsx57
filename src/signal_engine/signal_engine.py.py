"""
Enhanced Signal Engine for PocketBotX57.
Provides sophisticated signal generation using multiple strategies with adaptive learning.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import uuid
import json
import os
import logging

# Import strategies
from src.indicators.rsi_strategy import RSIStrategy
from src.indicators.macd_strategy import MACDStrategy
from src.indicators.bollinger_strategy import BollingerStrategy
from src.indicators.vwap_strategy import VWAPStrategy
from src.indicators.sma_cross_strategy import SMACrossStrategy
from src.indicators.pattern_recognition import PatternRecognition
from src.indicators.volume_analysis import VolumeAnalysis
from src.indicators.sentiment_strategy import SentimentStrategy

# Utilities
from src.utils.logger import get_logger
from src.utils.error_handler import SignalError, async_exception_handler

# Get logger
logger = get_logger("signal_engine")

class SignalEngine:
    """
    Enhanced Signal Engine with multi-strategy approach and machine learning optimization.
    Provides sophisticated trade signals with continuous adaptation from feedback.
    """
    
    def __init__(self, config_path: str = "config/signal_engine.json"):
        """
        Initialize the signal engine with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.strategies_config = self.config.get("strategies", {})
        
        # Initialize strategies
        self.strategies = {
            "rsi_strategy": RSIStrategy(self.strategies_config.get("rsi_strategy")),
            "macd_strategy": MACDStrategy(self.strategies_config.get("macd_strategy")),
            "bollinger_strategy": BollingerStrategy(self.strategies_config.get("bollinger_strategy")),
            "vwap_strategy": VWAPStrategy(self.strategies_config.get("vwap_strategy")),
            "sma_cross_strategy": SMACrossStrategy(self.strategies_config.get("sma_cross_strategy")),
            "pattern_recognition": PatternRecognition(self.strategies_config.get("pattern_recognition")),
            "volume_analysis": VolumeAnalysis(self.strategies_config.get("volume_analysis")),
            "sentiment_strategy": SentimentStrategy(self.strategies_config.get("sentiment_strategy"))
        }
        
        # Strategy weights (will be updated through learning)
        self.strategy_weights = {}
        self._initialize_weights()
        
        # Shadow strategies for experimental testing
        self.shadow_strategies = {}
        self._initialize_shadow_strategies()
        
        # Performance tracking
        self.signal_history = []
        self.strategy_performance = {}
        
        # Learning parameters
        self.learning_rate = self.config.get("learning_rate", 0.05)
        self.min_signals_for_learning = self.config.get("min_signals_for_learning", 10)
        
        # Signal thresholds
        self.confidence_threshold = self.config.get("confidence_threshold", 75)
        
        # Load history if available
        history_path = self.config.get("history_path", "data/signal_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    self.signal_history = json.load(f)
                logger.info(f"Loaded {len(self.signal_history)} historical signals")
            except Exception as e:
                logger.error(f"Error loading signal history: {str(e)}")
        
        logger.info("Signal Engine initialized with adaptive learning capabilities")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Configuration file {config_path} not found. Using defaults.")
                return {}
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def _initialize_weights(self):
        """Initialize strategy weights from config or defaults."""
        default_weights = {
            "rsi_strategy": 0.15,
            "macd_strategy": 0.20,
            "bollinger_strategy": 0.15,
            "vwap_strategy": 0.10,
            "sma_cross_strategy": 0.10,
            "pattern_recognition": 0.10,
            "volume_analysis": 0.10,
            "sentiment_strategy": 0.10
        }
        
        # Use config weights if available, otherwise use defaults
        weights = self.config.get("strategy_weights", default_weights)
        
        # Validate and normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            weights = default_weights
            total_weight = sum(weights.values())
        
        # Normalize to sum to 1.0
        self.strategy_weights = {
            strategy: weight / total_weight
            for strategy, weight in weights.items()
        }
    
    def _initialize_shadow_strategies(self):
        """Initialize experimental shadow strategies for testing."""
        # Shadow strategies are variants of main strategies with different parameters
        # They don't affect the main signal but are tracked for performance comparison
        
        self.shadow_strategies = {
            "aggressive_rsi": RSIStrategy({
                "period": 14,
                "overbought": 65,  # More aggressive thresholds
                "oversold": 35,
                "weight": 0.2
            }),
            "fast_macd": MACDStrategy({
                "fast_period": 8,  # Faster MACD for quicker signals
                "slow_period": 17,
                "signal_period": 9,
                "weight": 0.2
            }),
            "tight_bollinger": BollingerStrategy({
                "period": 20,
                "std_dev": 1.5,  # Tighter bands for more frequent signals
                "weight": 0.15
            }),
            "triple_cross": SMACrossStrategy({
                "periods": [5, 20, 50],  # Triple cross strategy
                "weight": 0.15
            })
        }
    
    def save_history(self):
        """Save signal history to file."""
        history_path = self.config.get("history_path", "data/signal_history.json")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        try:
            with open(history_path, 'w') as f:
                json.dump(self.signal_history, f)
            logger.info(f"Saved {len(self.signal_history)} signals to history")
        except Exception as e:
            logger.error(f"Error saving signal history: {str(e)}")
    
    def _preprocess_data(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess market data into pandas DataFrame format.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            if not market_data or "ohlc" not in market_data:
                raise SignalError("Missing OHLC data")
            
            # Convert to DataFrame if needed
            if isinstance(market_data["ohlc"], list):
                df = pd.DataFrame(market_data["ohlc"])
            elif isinstance(market_data["ohlc"], pd.DataFrame):
                df = market_data["ohlc"].copy()
            else:
                raise SignalError("Invalid OHLC data format")
            
            # Ensure required columns are present
            required_cols = ["timestamp", "open", "high", "low", "close"]
            if not all(col in df.columns for col in required_cols):
                raise SignalError(f"Missing required columns in market data: {required_cols}")
            
            # Set index if not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.set_index("timestamp", inplace=True)
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Add symbol if available
            if "symbol" in market_data:
                df["symbol"] = market_data["symbol"]
            
            return df
            
        except Exception as e:
            raise SignalError(f"Error preprocessing market data: {str(e)}")
    
    def _calculate_strategy_signals(self, data: pd.DataFrame, 
                                  include_shadow: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Calculate signals from all strategies.
        
        Args:
            data: Preprocessed market data
            include_shadow: Whether to include shadow strategies
            
        Returns:
            Dictionary with signals from each strategy
        """
        strategy_signals = {}
        
        # Process main strategies
        for name, strategy in self.strategies.items():
            try:
                if strategy.enabled:
                    signal = strategy.get_signal(data)
                    strategy_signals[name] = signal
            except Exception as e:
                logger.warning(f"Error in strategy {name}: {str(e)}")
                # Use neutral signal for failed strategies
                strategy_signals[name] = {
                    "direction": "NEUTRAL",
                    "confidence": 0,
                    "details": {"error": str(e)}
                }
        
        # Process shadow strategies if requested
        if include_shadow:
            for name, strategy in self.shadow_strategies.items():
                try:
                    signal = strategy.get_signal(data)
                    strategy_signals[f"shadow_{name}"] = signal
                except Exception as e:
                    logger.warning(f"Error in shadow strategy {name}: {str(e)}")
        
        return strategy_signals
    
    def _calculate_optimal_timeframe(self, data: pd.DataFrame, 
                                   signal_direction: str) -> int:
        """
        Calculate optimal trade timeframe based on market conditions.
        
        Args:
            data: Market data
            signal_direction: BUY or SELL
            
        Returns:
            Optimal timeframe in minutes
        """
        # Get volatility from ATR
        volatility = 0
        try:
            # Calculate ATR if possible
            tr1 = data['high'] - data['low']
            tr2 = abs(data['high'] - data['close'].shift())
            tr3 = abs(data['low'] - data['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            
            # Normalize as percentage of price
            current_price = data['close'].iloc[-1]
            volatility = (atr / current_price) * 100
        except Exception as e:
            logger.warning(f"Error calculating ATR: {str(e)}")
        
        # Base timeframe selection on volatility
        if volatility > 1.5:  # High volatility
            return 1  # 1 minute for high volatility
        elif volatility > 0.8:  # Medium volatility
            return 3  # 3 minutes
        else:  # Low volatility
            return 5  # 5 minutes for low volatility
    
    def generate_signal(self, market_data: Dict[str, Any], 
                       sentiment_data: Optional[Dict[str, Any]] = None,
                       include_shadow: bool = False) -> Dict[str, Any]:
        """
        Generate trading signal by combining multiple strategies.
        
        Args:
            market_data: Market data dictionary
            sentiment_data: Optional sentiment data
            include_shadow: Whether to include shadow strategies
            
        Returns:
            Signal dictionary with direction, confidence, and details
        """
        try:
            # Preprocess market data
            df = self._preprocess_data(market_data)
            
            # Add sentiment data to market data if available
            if sentiment_data:
                self.strategies["sentiment_strategy"].update_sentiment(sentiment_data)
            
            # Calculate signals from all strategies
            strategy_signals = self._calculate_strategy_signals(df, include_shadow)
            
            # Calculate weighted signal
            buy_confidence = 0
            sell_confidence = 0
            
            # Combine signals using weighted approach
            for strategy_name, signal in strategy_signals.items():
                # Skip shadow strategies for main signal
                if strategy_name.startswith("shadow_"):
                    continue
                
                # Get weight for this strategy
                weight = self.strategy_weights.get(strategy_name, 0.1)
                
                if signal["direction"] == "BUY":
                    buy_confidence += signal["confidence"] * weight
                elif signal["direction"] == "SELL":
                    sell_confidence += signal["confidence"] * weight
            
            # Determine final signal direction and confidence
            if buy_confidence > sell_confidence:
                direction = "BUY"
                confidence = buy_confidence
            elif sell_confidence > buy_confidence:
                direction = "SELL"
                confidence = sell_confidence
            else:
                direction = "NEUTRAL"
                confidence = 0
            
            # Calculate optimal timeframe
            timeframe = self._calculate_optimal_timeframe(df, direction)
            
            # Generate signal ID
            signal_id = str(uuid.uuid4())[:8]
            
            # Create signal object
            signal = {
                "id": signal_id,
                "timestamp": datetime.now().isoformat(),
                "symbol": market_data.get("symbol", "UNKNOWN"),
                "direction": direction,
                "confidence": confidence,
                "timeframe": timeframe,
                "price": df['close'].iloc[-1] if not df.empty else None,
                "strategy_signals": strategy_signals,
                "threshold_met": confidence >= self.confidence_threshold,
                "result": None  # Will be updated with feedback
            }
            
            # Add to history
            self.signal_history.append(signal)
            
            # Save periodically (every 10 signals)
            if len(self.signal_history) % 10 == 0:
                self.save_history()
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            raise SignalError(f"Signal generation failed: {str(e)}")
    
    def process_feedback(self, signal_id: str, result: bool) -> bool:
        """
        Process trade feedback to update strategy weights.
        
        Args:
            signal_id: Signal identifier
            result: True for win, False for loss
            
        Returns:
            Boolean indicating success
        """
        try:
            # Find the signal in history
            signal = None
            for s in self.signal_history:
                if s.get("id") == signal_id:
                    signal = s
                    break
            
            if not signal:
                logger.warning(f"Signal {signal_id} not found in history")
                return False
            
            # Update signal result
            signal["result"] = "win" if result else "loss"
            
            # Update strategy weights if we have enough signals
            if len([s for s in self.signal_history if s.get("result") is not None]) >= self.min_signals_for_learning:
                self._update_strategy_weights(signal, result)
            
            # Save history
            self.save_history()
            
            logger.info(f"Processed feedback for signal {signal_id}: {'win' if result else 'loss'}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return False
    
    def _update_strategy_weights(self, signal: Dict[str, Any], result: bool):
        """
        Update strategy weights based on feedback.
        
        Args:
            signal: Signal dictionary
            result: True for win, False for loss
        """
        # Extract strategy signals from the signal
        strategy_signals = signal.get("strategy_signals", {})
        
        # Update weights for each strategy based on whether it agreed with the overall signal
        adjustments = {}
        
        for strategy_name, strategy_signal in strategy_signals.items():
            # Skip shadow strategies
            if strategy_name.startswith("shadow_"):
                continue
                
            # Check if strategy agreed with overall signal
            agreed_with_signal = strategy_signal.get("direction") == signal.get("direction")
            
            # Calculate adjustment factor
            adjustment = 0
            if agreed_with_signal and result:
                # Strategy agreed and we won: Positive reinforcement
                adjustment = self.learning_rate
            elif agreed_with_signal and not result:
                # Strategy agreed but we lost: Negative reinforcement
                adjustment = -self.learning_rate
            elif not agreed_with_signal and not result:
                # Strategy disagreed and we lost: Positive reinforcement for disagreement
                adjustment = self.learning_rate * 0.5  # Less reinforcement for disagreement
            elif not agreed_with_signal and result:
                # Strategy disagreed but we won: Negative reinforcement for disagreement
                adjustment = -self.learning_rate * 0.5
            
            # Store adjustment for this strategy
            adjustments[strategy_name] = adjustment
            
            # Update strategy performance tracking
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = {
                    "total": 0,
                    "correct": 0,
                    "accuracy": 0
                }
            
            perf = self.strategy_performance[strategy_name]
            perf["total"] += 1
            
            if (agreed_with_signal and result) or (not agreed_with_signal and not result):
                perf["correct"] += 1
            
            perf["accuracy"] = perf["correct"] / perf["total"] if perf["total"] > 0 else 0
        
        # Apply adjustments to weights
        for strategy_name, adjustment in adjustments.items():
            if strategy_name in self.strategy_weights:
                self.strategy_weights[strategy_name] += adjustment
        
        # Ensure weights are positive
        for strategy_name in self.strategy_weights:
            self.strategy_weights[strategy_name] = max(0.01, self.strategy_weights[strategy_name])
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.strategy_weights.values())
        for strategy_name in self.strategy_weights:
            self.strategy_weights[strategy_name] /= total_weight
        
        logger.info(f"Updated strategy weights: {self.strategy_weights}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the signal engine.
        
        Returns:
            Dictionary with performance statistics
        """
        stats = {
            "total_signals": len(self.signal_history),
            "evaluated_signals": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0,
            "strategy_performance": self.strategy_performance,
            "strategy_weights": self.strategy_weights,
            "recent_signals": []
        }
        
        # Process signals with feedback
        for signal in self.signal_history:
            if signal.get("result") is not None:
                stats["evaluated_signals"] += 1
                
                if signal["result"] == "win":
                    stats["win_count"] += 1
                else:
                    stats["loss_count"] += 1
        
        # Calculate win rate
        if stats["evaluated_signals"] > 0:
            stats["win_rate"] = (stats["win_count"] / stats["evaluated_signals"]) * 100
        
        # Get recent signals
        stats["recent_signals"] = sorted(
            self.signal_history[-10:],
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        
        return stats
    
    def evaluate_shadow_strategies(self) -> Dict[str, Any]:
        """
        Evaluate performance of shadow strategies.
        
        Returns:
            Dictionary with shadow strategy performance
        """
        shadow_performance = {}
        
        # Loop through signals with feedback
        for signal in self.signal_history:
            if signal.get("result") is None:
                continue
                
            strategy_signals = signal.get("strategy_signals", {})
            actual_result = signal.get("result") == "win"
            
            # Evaluate each shadow strategy
            for strategy_name, strategy_signal in strategy_signals.items():
                if not strategy_name.startswith("shadow_"):
                    continue
                
                # Initialize performance tracking for this strategy
                if strategy_name not in shadow_performance:
                    shadow_performance[strategy_name] = {
                        "total": 0,
                        "correct": 0,
                        "accuracy": 0,
                        "avg_confidence": 0
                    }
                
                perf = shadow_performance[strategy_name]
                perf["total"] += 1
                
                # Check if shadow strategy was correct
                shadow_direction = strategy_signal.get("direction", "NEUTRAL")
                signal_direction = signal.get("direction", "NEUTRAL")
                
                # Shadow correct if it matched signal direction and signal was right
                # or if it disagreed with signal direction and signal was wrong
                shadow_agreed = shadow_direction == signal_direction
                
                if (shadow_agreed and actual_result) or (not shadow_agreed and not actual_result):
                    perf["correct"] += 1
                
                # Update accuracy and confidence
                perf["accuracy"] = perf["correct"] / perf["total"] if perf["total"] > 0 else 0
                perf["avg_confidence"] += strategy_signal.get("confidence", 0)
                perf["avg_confidence"] /= perf["total"]
        
        return shadow_performance
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get overall status of the signal engine.
        
        Returns:
            Dictionary with engine status
        """
        return {
            "strategies": list(self.strategies.keys()),
            "weights": self.strategy_weights,
            "shadow_strategies": list(self.shadow_strategies.keys()),
            "learning_rate": self.learning_rate,
            "confidence_threshold": self.confidence_threshold,
            "history_length": len(self.signal_history),
            "last_updated": datetime.now().isoformat()
        }