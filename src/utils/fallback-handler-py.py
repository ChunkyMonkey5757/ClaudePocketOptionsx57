"""
Fallback Handler for PocketBotX57.
Provides last-resort data handling when all API sources fail.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

from src.utils.logger import get_logger
from src.utils.error_handler import APIError, async_exception_handler

# Get logger
logger = get_logger("fallback_handler")

@async_exception_handler
async def generate_synthetic_data(symbol: str, interval: str = "1m", 
                                limit: int = 100) -> Dict[str, Any]:
    """
    Generate synthetic market data when all API sources fail.
    This is a last resort to keep the bot functioning.
    
    Args:
        symbol (str): Asset symbol (e.g., BTC, ETH)
        interval (str, optional): Time interval. Defaults to "1m".
        limit (int, optional): Number of data points. Defaults to 100.
        
    Returns:
        Dict[str, Any]: Synthetic market data
    """
    logger.warning(f"Generating synthetic data for {symbol} as fallback")
    
    # Create a base price (random but plausible)
    base_prices = {
        "BTC": 50000.0,
        "ETH": 3000.0,
        "BNB": 500.0,
        "SOL": 150.0,
        "XRP": 0.5,
        "ADA": 0.4,
        "DOGE": 0.1,
        "MATIC": 0.9,
        "DOT": 10.0,
        "AVAX": 20.0,
        "LINK": 15.0,
        "UNI": 5.0,
        "ATOM": 8.0
    }
    
    # Use a predefined base price if available, otherwise use a generic one
    base_price = base_prices.get(symbol.upper(), 100.0)
    
    # Generate timestamps
    end_time = datetime.now()
    
    # Determine time delta based on interval
    if interval == "1m":
        delta = timedelta(minutes=1)
    elif interval == "5m":
        delta = timedelta(minutes=5)
    elif interval == "15m":
        delta = timedelta(minutes=15)
    elif interval == "30m":
        delta = timedelta(minutes=30)
    elif interval == "1h":
        delta = timedelta(hours=1)
    elif interval == "4h":
        delta = timedelta(hours=4)
    elif interval == "1d":
        delta = timedelta(days=1)
    else:
        delta = timedelta(minutes=1)
    
    # Generate timestamps in descending order (newest first)
    timestamps = [end_time - (delta * i) for i in range(limit)]
    timestamps.reverse()  # Now in ascending order (oldest first)
    
    # Generate synthetic price data with some randomness and trend
    # Use a random walk with drift
    np.random.seed(int(datetime.now().timestamp() % 100000))  # Semi-random seed
    
    # Parameters for random walk
    volatility = base_price * 0.01  # 1% volatility
    drift = base_price * 0.0001  # Small upward drift
    
    # Generate prices with random walk
    returns = np.random.normal(drift, volatility, limit)
    prices = np.exp(np.cumsum(returns)) * base_price
    
    # Generate OHLC data
    ohlc_data = []
    for i in range(limit):
        # Add some randomness to open, high, low
        price = prices[i]
        high_low_range = price * 0.005  # 0.5% range
        
        open_price = price * (1 + np.random.uniform(-0.002, 0.002))
        high_price = max(price, open_price) + np.random.uniform(0, high_low_range)
        low_price = min(price, open_price) - np.random.uniform(0, high_low_range)
        close_price = price
        
        # Generate a plausible volume
        volume = base_price * 10 * np.random.uniform(0.5, 1.5)
        
        ohlc_data.append({
            "timestamp": timestamps[i],
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume
        })
    
    # Construct result
    result = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "exchange": "synthetic",
        "timestamp": datetime.now().isoformat(),
        "ohlc": ohlc_data,
        "price": ohlc_data[-1]["close"],
        "bid": ohlc_data[-1]["close"] * 0.999,  # Slightly lower
        "ask": ohlc_data[-1]["close"] * 1.001,  # Slightly higher
        "volume_24h": sum(data["volume"] for data in ohlc_data[-24:]) if len(ohlc_data) >= 24 else sum(data["volume"] for data in ohlc_data),
        "percent_change_24h": ((ohlc_data[-1]["close"] / ohlc_data[0]["close"]) - 1) * 100 if len(ohlc_data) > 0 else 0,
        "synthetic": True  # Flag to indicate this is synthetic data
    }
    
    return result

@async_exception_handler
async def fetch_last_known_price(api_manager: Any, symbol: str) -> float:
    """
    Fetch the last known price from cache.
    
    Args:
        api_manager: API Manager instance
        symbol (str): Asset symbol
        
    Returns:
        float: Last known price or a synthetic price
    """
    # Check cache for any recent price data
    for api_name in api_manager.api_priority:
        cache_key = api_manager._get_cache_key(api_name, f"price_{symbol}", {})
        if api_manager._is_cache_valid(cache_key):
            return api_manager.cache[cache_key]
    
    # If no cache is available, generate a synthetic price
    base_prices = {
        "BTC": 50000.0,
        "ETH": 3000.0,
        "BNB": 500.0,
        "SOL": 150.0,
        "XRP": 0.5,
        "ADA": 0.4,
        "DOGE": 0.1,
        "MATIC": 0.9,
        "DOT": 10.0,
        "AVAX": 20.0,
        "LINK": 15.0,
        "UNI": 5.0,
        "ATOM": 8.0
    }
    
    return base_prices.get(symbol.upper(), 100.0)

@async_exception_handler
async def get_default_assets() -> List[str]:
    """
    Get a default list of assets when all API sources fail.
    
    Returns:
        List[str]: Default list of asset symbols
    """
    return [
        "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", 
        "MATIC", "DOT", "AVAX", "LINK", "UNI", "ATOM"
    ]