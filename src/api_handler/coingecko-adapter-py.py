"""
CoinGecko Adapter for PocketBotX57.
Provides cryptocurrency market data using the CoinGecko API.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Union
import time
from datetime import datetime, timedelta

from src.utils.logger import get_logger
from src.utils.error_handler import APIError, async_exception_handler

# Get logger
logger = get_logger("coingecko_adapter")

# CoinGecko API base URL
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# Mapping of intervals to CoinGecko days parameter
DAYS_MAP = {
    "1m": 1,     # 1 day data with minutely data points
    "3m": 1,
    "5m": 1,
    "15m": 1,
    "30m": 2,    # 2 days data with more granular data
    "1h": 7,     # 7 days data with hourly data points
    "2h": 14,
    "4h": 30,
    "6h": 30,
    "12h": 60,
    "1d": 90,
}

@async_exception_handler
async def fetch_market_data(api_manager: Any, symbol: str, interval: str = "1m", 
                          limit: int = 100) -> Dict[str, Any]:
    """
    Fetch market data using CoinGecko API.
    
    Args:
        api_manager: API Manager instance
        symbol (str): Asset symbol (e.g., BTC, ETH)
        interval (str, optional): Time interval. Defaults to "1m".
        limit (int, optional): Number of data points. Defaults to 100.
        
    Returns:
        Dict[str, Any]: Market data
        
    Raises:
        APIError: If the request fails
    """
    # Convert symbol to CoinGecko format (lowercase)
    coin_id = symbol.lower()
    
    # Get days parameter based on interval
    days = DAYS_MAP.get(interval, 1)
    
    try:
        # First, get coin info to verify it exists
        coin_info_url = f"{COINGECKO_API_URL}/coins/{coin_id}"
        coin_info = await api_manager.make_request(
            api_name="coingecko",
            method="GET",
            url=coin_info_url,
            cache_expiry=300  # Cache for 5 minutes
        )
        
        # Get market data
        market_chart_url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily" if interval == "1d" else None
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        market_data = await api_manager.make_request(
            api_name="coingecko",
            method="GET",
            url=market_chart_url,
            params=params,
            cache_expiry=60  # Cache for 1 minute
        )
        
        # Process OHLC data - CoinGecko only provides prices, volumes, and market caps
        # We need to create a pandas DataFrame
        prices = market_data.get("prices", [])
        volumes = market_data.get("total_volumes", [])
        
        # Create DataFrame
        df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
        
        df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
        df_volumes['timestamp'] = pd.to_datetime(df_volumes['timestamp'], unit='ms')
        
        # Merge DataFrames
        df = pd.merge(df_prices, df_volumes, on='timestamp')
        
        # Resample to desired interval if needed
        # Note: CoinGecko doesn't provide exact OHLC data, so we're approximating
        df = df.sort_values('timestamp')
        
        if interval == "1m":
            df = df.resample('1min', on='timestamp').agg({'price': 'ohlc', 'volume': 'sum'})
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.reset_index()
        elif interval == "5m":
            df = df.resample('5min', on='timestamp').agg({'price': 'ohlc', 'volume': 'sum'})
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.reset_index()
        elif interval == "15m":
            df = df.resample('15min', on='timestamp').agg({'price': 'ohlc', 'volume': 'sum'})
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.reset_index()
        elif interval == "30m":
            df = df.resample('30min', on='timestamp').agg({'price': 'ohlc', 'volume': 'sum'})
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.reset_index()
        elif interval == "1h":
            df = df.resample('1h', on='timestamp').agg({'price': 'ohlc', 'volume': 'sum'})
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.reset_index()
        
        # Limit to the requested number of data points
        df = df.tail(limit)
        
        # Get current market data
        coin_market_url = f"{COINGECKO_API_URL}/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": coin_id,
            "order": "market_cap_desc",
            "per_page": 1,
            "page": 1,
            "sparkline": False,
            "locale": "en"
        }
        
        coin_market = await api_manager.make_request(
            api_name="coingecko",
            method="GET",
            url=coin_market_url,
            params=params,
            cache_expiry=60  # Cache for 1 minute
        )
        
        # Extract market data
        market_info = coin_market[0] if coin_market else {}
        
        # Construct result
        result = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "exchange": "coingecko",
            "timestamp": datetime.now().isoformat(),
            "ohlc": df.to_dict('records'),
            "price": market_info.get("current_price", 0),
            "bid": None,  # CoinGecko doesn't provide bid/ask
            "ask": None,
            "volume_24h": market_info.get("total_volume", 0),
            "percent_change_24h": market_info.get("price_change_percentage_24h", 0),
            "market_cap": market_info.get("market_cap", 0),
            "high_24h": market_info.get("high_24h", 0),
            "low_24h": market_info.get("low_24h", 0),
        }
        
        return result
    
    except Exception as e:
        raise APIError(f"CoinGecko fetch market data failed for {symbol}: {str(e)}")

@async_exception_handler
async def fetch_price(api_manager: Any, symbol: str) -> float:
    """
    Fetch current price for a symbol using CoinGecko.
    
    Args:
        api_manager: API Manager instance
        symbol (str): Asset symbol
        
    Returns:
        float: Current price
        
    Raises:
        APIError: If the request fails
    """
    try:
        # Get simple price
        price_url = f"{COINGECKO_API_URL}/simple/price"
        params = {
            "ids": symbol.lower(),
            "vs_currencies": "usd"
        }
        
        price_data = await api_manager.make_request(
            api_name="coingecko",
            method="GET",
            url=price_url,
            params=params,
            cache_expiry=30  # Cache for 30 seconds
        )
        
        # Extract price
        price = price_data.get(symbol.lower(), {}).get("usd")
        
        if price is None:
            raise APIError(f"CoinGecko price data not found for {symbol}")
        
        return float(price)
    
    except Exception as e:
        raise APIError(f"CoinGecko fetch price failed for {symbol}: {str(e)}")

@async_exception_handler
async def get_available_assets(api_manager: Any) -> List[str]:
    """
    Get list of available assets using CoinGecko.
    
    Args:
        api_manager: API Manager instance
        
    Returns:
        List[str]: List of available asset symbols
        
    Raises:
        APIError: If the request fails
    """
    try:
        # Get top 100 coins by market cap
        coins_url = f"{COINGECKO_API_URL}/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": False,
            "locale": "en"
        }
        
        coins = await api_manager.make_request(
            api_name="coingecko",
            method="GET",
            url=coins_url,
            params=params,
            cache_expiry=300  # Cache for 5 minutes
        )
        
        # Extract symbols
        symbols = [coin.get("symbol", "").upper() for coin in coins]
        
        # Remove duplicates and sort
        symbols = sorted(list(set(symbols)))
        
        return symbols
    
    except Exception as e:
        raise APIError(f"CoinGecko get available assets failed: {str(e)}")