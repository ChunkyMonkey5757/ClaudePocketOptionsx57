"""
Enhanced Signal Engine for PocketBotX57.
Provides sophisticated signal generation using multiple strategies with adaptive learning.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import uuid

import pandas as pd
import numpy as np

from src.indicators.indicator_base import IndicatorBase
from src.indicators.rsi import RSIStrategy
from src.indicators.macd import MACDStrategy
from src.indicators.bollinger import BollingerStrategy
from src.indicators.vwap import VWAPStrategy
from src.indicators.sma_cross import SMACrossStrategy
from src.indicators.pattern import PatternRecognition
from src.indicators.volume import VolumeAnalysis

from src.signal_engine.filter import SignalFilter
from src.utils.logger import get_logger
from src.utils.error_handler import SignalError

logger = get_logger(__name__)

class SignalEngine:
    """
    Enhanced Signal Engine with multi-strategy approach and adaptive learning.
    Provides sophisticated trade signals with continuous improvement from feedback.
    """
    
    def __init__(self, config_path: str = "config/signal_engine.json"):
        """Initialize the signal engine with configuration."""
        self.config = self._load_config(config_path)
        
        # Initialize strategies
        self.strategies = {
            "rsi": RSIStrategy(self.config.get("strategies", {}).get("rsi", {})),
            "macd": MACDStrategy(self.config.get("strategies", {}).get("macd", {})),
            "bollinger": BollingerStrategy(self.config.get("strategies", {}).get("bollinger", {})),
            "vwap": VWAPStrategy(self.config.get("strategies", {}).get("vwap", {})),
            "sma_cross": SMACrossStrategy(self.config.get("strategies", {}).get("sma_cross", {})),
            "pattern": PatternRecognition(self.config.get("strategies", {}).get("pattern", {})),
            "volume": VolumeAnalysis(self.config.get("strategies", {}).get("volume", {})),
        }
        
        # Shadow strategies for testing alternatives
        self.shadow_strategies = self._init_shadow_strategies()
        
        # Strategy weights (will be adjusted by learning)
        self.strategy_weights = self._initialize_weights()
        
        # Signal filter
        self.signal_filter = SignalFilter(self.config.get("filter", {}))
        
        # Signal history for learning
        self.signal_history = []
        self.feedback_history = []
        
        # Load historical data
        self._load_history()
        
        # Learning parameters
        self.learning_rate = self.config.get("learning_rate", 0.05)
        self.min_signals_for_learning = self.config.get("min_signals_for_learning", 10)
        
        logger.info("Signal Engine initialized with %d strategies", len(self.strategies))
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning("Configuration file %s not found. Using defaults.", config_path)
                return {}
        except Exception as e:
            logger.error("Error loading config: %s", str(e))
            return {}
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize strategy weights from config or defaults."""
        default_weights = {
            "rsi": 0.15,
            "macd": 0.20,
            "bollinger": 0.15,
            "vwap": 0.10,
            "sma_cross": 0.10,
            "pattern": 0.15,
            "volume": 0.15,
        }
        
        weights = self.config.get("strategy_weights", default_weights)
        
        # Normalize to sum to 1.0
        total = sum(weights.values())
        if total <= 0:
            return default_weights
            
        return {k: v/total for k, v in weights.items()}
    
    def _init_shadow_strategies(self) -> Dict[str, IndicatorBase]:
        """Initialize experimental shadow strategies for testing."""
        shadow_strategies = {}
        
        # Create variants with different parameters
        shadow_strategies["aggressive_rsi"] = RSIStrategy({
            "name": "aggressive_rsi",
            "period": 10,
            "overbought": 65,
            "oversold": 35,
        })
        
        shadow_strategies["fast_macd"] = MACDStrategy({
            "name": "fast_macd",
            "fast_period": 8,
            "slow_period": 17,
            "signal_period": 9,
        })
        
        shadow_strategies["tight_bollinger"] = BollingerStrategy({
            "name": "tight_bollinger",
            "period": 20,
            "std_dev": 1.5,
        })
        
        return shadow_strategies
    
    def _load_history(self) -> None:
        """Load signal and feedback history from files."""
        signal_path = self.config.get("signal_history_path", "data/signal_history.json")
        feedback_path = self.config.get("feedback_history_path", "data/feedback_history.json")
        
        try:
            if os.path.exists(signal_path):
                with open(signal_path, 'r') as f:
                    self.signal_history = json.load(f)
                logger.info("Loaded %d historical signals", len(self.signal_history))
                
            if os.path.exists(feedback_path):
                with open(feedback_path, 'r') as f:
                    self.feedback_history = json.load(f)
                logger.info("Loaded %d feedback entries", len(self.feedback_history))
        except Exception as e:
            logger.error("Error loading history: %s", str(e))
    
    def save_history(self) -> None:
        """Save signal and feedback history to files."""
        signal_path = self.config.get("signal_history_path", "data/signal_history.json")
        feedback_path = self.config.get("feedback_history_path", "data/feedback_history.json")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(signal_path), exist_ok=True)
        os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
        
        try:
            with open(signal_path, 'w') as f:
                json.dump(self.signal_history, f)
                
            with open(feedback_path, 'w') as f:
                json.dump(self.feedback_history, f)
                
            logger.info("Saved signal and feedback history")
        except Exception as e:
            logger.error("Error saving history: %s", str(e))
    
    async def generate_signal(self, 
                             market_data: Dict[str, Any], 
                             sentiment_data: Optional[Dict[str, Any]] = None,
                             economic_events: Optional[List[Dict[str, Any]]] = None,
                             include_shadow: bool = False) -> Dict[str, Any]:
        """
        Generate trading signal by combining multiple strategies.
        
        Args:
            market_data: Market data dictionary
            sentiment_data: Optional sentiment data
            economic_events: Optional economic calendar events
            include_shadow: Whether to include shadow strategies
            
        Returns:
            Signal dictionary with direction, confidence, and details
        """
        try:
            # Validate market data
            if not market_data or "ohlc" not in market_data:
                raise SignalError("Invalid market data")
            
            # Convert to DataFrame if necessary
            df = self._prepare_dataframe(market_data)
            
            # Get symbol
            symbol = market_data.get("symbol", "UNKNOWN")
            
            # Process all strategies in parallel
            strategy_signals = await self._process_strategies(df, include_shadow)
            
            # Combine signals using weighted approach
            combined_signal = self._combine_signals(strategy_signals, symbol)
            
            # Filter signal based on various criteria
            filter_result = self.signal_filter.filter_signal(
                combined_signal, 
                market_data,
                economic_events
            )
            
            # Apply filter results
            combined_signal["confidence"] = filter_result["adjusted_confidence"]
            combined_signal["passed_filter"] = filter_result["passed"]
            combined_signal["filter_reasons"] = filter_result.get("reasons", [])
            
            # Generate signal ID
            signal_id = str(uuid.uuid4())[:8]
            
            # Determine optimal timeframe
            timeframe = self._calculate_optimal_timeframe(df, combined_signal["direction"])
            
            # Create final signal object
            signal = {
                "id": signal_id,
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "direction": combined_signal["direction"],
                "confidence": combined_signal["confidence"],
                "timeframe": timeframe,
                "price": df['close'].iloc[-1],
                "strategy_signals": strategy_signals,
                "shadow_signals": combined_signal.get("shadow_signals", {}),
                "passed_filter": combined_signal["passed_filter"],
                "filter_reasons": combined_signal["filter_reasons"],
                "indicators": combined_signal.get("indicators", {}),
                "result": None  # Will be updated with feedback
            }
            
            # Add to history
            self.signal_history.append(signal)
            
            # Save periodically (every 10 signals)
            if len(self.signal_history) % 10 == 0:
                self.save_history()
            
            return signal
            
        except Exception as e:
            logger.error("Error generating signal: %s", str(e))
            raise SignalError(f"Signal generation failed: {str(e)}")
    
    def _prepare_dataframe(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert market data to pandas DataFrame."""
        if isinstance(market_data["ohlc"], pd.DataFrame):
            return market_data["ohlc"]
        
        df = pd.DataFrame(market_data["ohlc"])
        
        # Set datetime index if not already
        if not isinstance(df.index, pd.DatetimeIndex) and 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    async def _process_strategies(self, 
                                df: pd.DataFrame, 
                                include_shadow: bool = False) -> Dict[str, Dict[str, Any]]:
        """Process all strategies in parallel."""
        strategy_signals = {}
        
        # Create tasks for all strategies
        tasks = []
        for name, strategy in self.strategies.items():
            if strategy.enabled:
                tasks.append(self._execute_strategy(name, strategy, df))
        
        # Add shadow strategies if requested
        if include_shadow:
            for name, strategy in self.shadow_strategies.items():
                if strategy.enabled:
                    tasks.append(self._execute_strategy(name, strategy, df, is_shadow=True))
        
        # Run all strategies in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for name, result in results:
            if isinstance(result, Exception):
                logger.warning("Strategy %s failed: %s", name, str(result))
                # Use neutral signal for failed strategies
                strategy_signals[name] = {
                    "direction": "NEUTRAL",
                    "confidence": 0,
                    "details": {"error": str(result)}
                }
            else:
                strategy_signals[name] = result
        
        return strategy_signals
    
    async def _execute_strategy(self, 
                              name: str, 
                              strategy: IndicatorBase, 
                              df: pd.DataFrame,
                              is_shadow: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Execute a strategy and return its signal."""
        try:
            signal = await strategy.get_signal(df)
            return (name, signal)
        except Exception as e:
            return (name, e)
    
    def _combine_signals(self, 
                       strategy_signals: Dict[str, Dict[str, Any]],
                       symbol: str) -> Dict[str, Any]:
        """Combine signals from all strategies using weighted approach."""
        buy_confidence = 0
        sell_confidence = 0
        indicators = {}
        
        # Separate standard and shadow strategies
        standard_signals = {k: v for k, v in strategy_signals.items() 
                           if k in self.strategies}
        shadow_signals = {k: v for k, v in strategy_signals.items() 
                         if k in self.shadow_strategies}
        
        # Process standard strategies
        for strategy_name, signal in standard_signals.items():
            # Get weight for this strategy
            weight = self.strategy_weights.get(strategy_name, 0.1)
            
            if signal["direction"] == "BUY":
                buy_confidence += signal["confidence"] * weight
            elif signal["direction"] == "SELL":
                sell_confidence += signal["confidence"] * weight
            
            # Collect indicator values
            if "indicators" in signal:
                indicators.update(signal["indicators"])
        
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
        
        return {
            "direction": direction,
            "confidence": confidence,
            "indicators": indicators,
            "shadow_signals": shadow_signals,
            "passed_filter": True,  # Will be updated by filter
            "filter_reasons": []
        }
    
    def _calculate_optimal_timeframe(self, df: pd.DataFrame, direction: str) -> int:
        """Calculate optimal trade timeframe based on volatility."""
        try:
            # Calculate Average True Range (ATR)
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            
            # Normalize as percentage of price
            current_price = df['close'].iloc[-1]
            volatility = (atr / current_price) * 100
            
            # Determine timeframe based on volatility
            if volatility > 1.5:
                return 1  # 1 minute for high volatility
            elif volatility > 0.8:
                return 3  # 3 minutes for medium volatility
            else:
                return 5  # 5 minutes for low volatility
        except Exception as e:
            logger.warning("Error calculating optimal timeframe: %s", str(e))
            return 3  # Default to 3 minutes
    
    async def process_feedback(self, signal_id: str, result: bool) -> bool:
        """
        Process user feedback on a signal (win/loss).
        
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
                logger.warning("Signal %s not found in history", signal_id)
                return False
            
            # Update signal result
            signal["result"] = "win" if result else "loss"
            
            # Record feedback
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "signal_id": signal_id,
                "result": "win" if result else "loss",
                "signal_data": signal
            }
            
            self.feedback_history.append(feedback_entry)
            
            # Update strategy weights
            if len(self.feedback_history) >= self.min_signals_for_learning:
                await self._update_strategy_weights(signal, result)
            
            # Save history
            self.save_history()
            
            logger.info("Processed feedback for signal %s: %s", 
                      signal_id, "win" if result else "loss")
            
            return True
            
        except Exception as e:
            logger.error("Error processing feedback: %s", str(e))
            return False
    
    async def _update_strategy_weights(self, signal: Dict[str, Any], result: bool) -> None:
        """Update strategy weights based on feedback."""
        # Extract strategy signals
        strategy_signals = signal.get("strategy_signals", {})
        
        # Calculate adjustments
        adjustments = {}
        for strategy_name, strategy_signal in strategy_signals.items():
            # Skip shadow strategies
            if strategy_name in self.shadow_strategies:
                continue
                
            # Check if strategy agreed with overall signal
            agreed = strategy_signal.get("direction") == signal.get("direction")
            
            # Calculate adjustment
            if agreed and result:
                # Strategy agreed and we won: positive reinforcement
                adjustment = self.learning_rate
            elif agreed and not result:
                # Strategy agreed but we lost: negative reinforcement
                adjustment = -self.learning_rate
            elif not agreed and not result:
                # Strategy disagreed and we lost: positive for disagreement
                adjustment = self.learning_rate * 0.5
            elif not agreed and result:
                # Strategy disagreed but we won: negative for disagreement
                adjustment = -self.learning_rate * 0.5
            else:
                adjustment = 0
            
            adjustments[strategy_name] = adjustment
        
        # Apply adjustments
        for strategy_name, adjustment in adjustments.items():
            if strategy_name in self.strategy_weights:
                self.strategy_weights[strategy_name] += adjustment
                
                # Ensure weight is positive
                self.strategy_weights[strategy_name] = max(0.01, self.strategy_weights[strategy_name])
        
        # Normalize weights to sum to 1.0
        total = sum(self.strategy_weights.values())
        for strategy_name in self.strategy_weights:
            self.strategy_weights[strategy_name] /= total
        
        logger.info("Updated strategy weights: %s", self.strategy_weights)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the signal engine."""
        stats = {
            "total_signals": len(self.signal_history),
            "signals_with_feedback": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0.0,
            "strategy_performance": {},
            "by_asset": {},
            "by_timeframe": {},
            "recent_signals": []
        }
        
        # Calculate overall stats
        for signal in self.signal_history:
            if signal.get("result") is not None:
                stats["signals_with_feedback"] += 1
                
                if signal["result"] == "win":
                    stats["win_count"] += 1
                else:
                    stats["loss_count"] += 1
                
                # Track by asset
                symbol = signal.get("symbol", "UNKNOWN")
                if symbol not in stats["by_asset"]:
                    stats["by_asset"][symbol] = {"wins": 0, "losses": 0}
                
                if signal["result"] == "win":
                    stats["by_asset"][symbol]["wins"] += 1
                else:
                    stats["by_asset"][symbol]["losses"] += 1
                
                # Track by timeframe
                timeframe = signal.get("timeframe", 0)
                tf_key = f"{timeframe}m"
                if tf_key not in stats["by_timeframe"]:
                    stats["by_timeframe"][tf_key] = {"wins": 0, "losses": 0}
                
                if signal["result"] == "win":
                    stats["by_timeframe"][tf_key]["wins"] += 1
                else:
                    stats["by_timeframe"][tf_key]["losses"] += 1
        
        # Calculate win rate
        if stats["signals_with_feedback"] > 0:
            stats["win_rate"] = (stats["win_count"] / stats["signals_with_feedback"]) * 100
        
        # Calculate individual strategy performance
        strategy_performance = {}
        for signal in self.signal_history:
            if signal.get("result") is None:
                continue
                
            result = signal["result"] == "win"
            strategy_signals = signal.get("strategy_signals", {})
            
            for strategy_name, strategy_signal in strategy_signals.items():
                if strategy_name not in strategy_performance:
                    strategy_performance[strategy_name] = {
                        "total": 0,
                        "correct": 0,
                        "win_rate": 0.0
                    }
                
                agreed = strategy_signal.get("direction") == signal.get("direction")
                strategy_performance[strategy_name]["total"] += 1
                
                if (agreed and result) or (not agreed and not result):
                    strategy_performance[strategy_name]["correct"] += 1
        
        # Calculate strategy win rates
        for strategy, perf in strategy_performance.items():
            if perf["total"] > 0:
                perf["win_rate"] = (perf["correct"] / perf["total"]) * 100
        
        stats["strategy_performance"] = strategy_performance
        
        # Get recent signals
        stats["recent_signals"] = sorted(
            self.signal_history[-10:],
            key=lambda s: s.get("timestamp", ""),
            reverse=True
        )
        
        return stats