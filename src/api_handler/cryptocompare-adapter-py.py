"""
CryptoCompare Adapter for PocketBotX57.
Provides cryptocurrency market data using the CryptoCompare API.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

from src.utils.logger import get_logger
from src.utils.error_handler import APIError, async_exception_handler

# Get logger
logger = get_logger("cryptocompare_adapter")

# CryptoCompare API base URL
CRYPTOCOMPARE_API_URL = "https://min-api.cryptocompare.com/data"

# Mapping of intervals to CryptoCompare intervals
TIMEFRAME_MAP = {
    "1m": "histominute",
    "3m": "histominute",
    "5m": "histominute",
    "15m": "histominute",
    "30m": "histominute",
    "1h": "histohour",
    "2h": "histohour",
    "4h": "histohour",
    "6h": "histohour",
    "12h": "histohour",
    "1d": "histoday",
}

# Mapping of intervals to limit and aggregation values
LIMIT_AGGREGATION_MAP = {
    "1m": {"limit": 1440, "aggregation": 1},
    "3m": {"limit": 480, "aggregation": 3},
    "5m": {"limit": 288, "aggregation": 5},
    "15m": {"limit": 96, "aggregation": 15},
    "30m": {"limit": 48, "aggregation": 30},
    "1h": {"limit": 168, "aggregation": 1},
    "2h": {"limit": 84, "aggregation": 2},
    "4h": {"limit": 42, "aggregation": 4},
    "6h": {"limit": 28, "aggregation": 6},
    "12h": {"limit": 14, "aggregation": 12},
    "1d": {"limit": 30, "aggregation": 1},
}

@async_exception_handler
async def fetch_market_data(api_manager: Any, symbol: str, interval: str = "1m", 
                          limit: int = 100) -> Dict[str, Any]:
    """
    Fetch market data using CryptoCompare API.
    
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
    try:
        # Get appropriate endpoint and parameters
        timeframe = TIMEFRAME_MAP.get(interval, "histominute")
        limit_aggregation = LIMIT_AGGREGATION_MAP.get(interval, {"limit": 100, "aggregation": 1})
        
        # Adjust limit and aggregation
        adjusted_limit = min(limit, limit_aggregation["limit"])
        aggregation = limit_aggregation["aggregation"]
        
        # Get API key from config if available
        api_key = api_manager.api_keys.get("cryptocompare", "")
        headers = {"authorization": f"Apikey {api_key}"} if api_key else None
        
        # Fetch historical data
        histo_url = f"{CRYPTOCOMPARE_API_URL}/{timeframe}"
        params = {
            "fsym": symbol.upper(),
            "tsym": "USD",
            "limit": adjusted_limit,
            "aggregate": aggregation,
            "e": "CCCAGG"  # Cryptocurrency Aggregate Index
        }
        
        histo_data = await api_manager.make_request(
            api_name="cryptocompare",
            method="GET",
            url=histo_url,
            params=params,
            headers=headers,
            cache_expiry=60  # Cache for 1 minute
        )
        
        # Check response
        if not histo_data.get("Data"):
            raise APIError(f"CryptoCompare historical data not found for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(histo_data["Data"])
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volumefrom": "volume"
        })
        
        # Get current price data
        price_url = f"{CRYPTOCOMPARE_API_URL}/price"
        price_params = {
            "fsym": symbol.upper(),
            "tsyms": "USD",
            "e": "CCCAGG"
        }
        
        price_data = await api_manager.make_request(
            api_name="cryptocompare",
            method="GET",
            url=price_url,
            params=price_params,
            headers=headers,
            cache_expiry=30  # Cache for 30 seconds
        )
        
        # Get 24h price data
        price_24h_url = f"{CRYPTOCOMPARE_API_URL}/pricemultifull"
        price_24h_params = {
            "fsyms": symbol.upper(),
            "tsyms": "USD",
            "e": "CCCAGG"
        }
        
        price_24h_data = await api_manager.make_request(
            api_name="cryptocompare",
            method="GET",
            url=price_24h_url,
            params=price_24h_params,
            headers=headers,
            cache_expiry=60  # Cache for 1 minute
        )
        
        # Extract price data
        raw_data = price_24h_data.get("RAW", {}).get(symbol.upper(), {}).get("USD", {})
        display_data = price_24h_data.get("DISPLAY", {}).get(symbol.upper(), {}).get("USD", {})
        
        # Construct result
        result = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "exchange": "cryptocompare",
            "timestamp": datetime.now().isoformat(),
            "ohlc": df[["timestamp", "open", "high", "low", "close", "volume"]].to_dict('records'),
            "price": price_data.get("USD", 0),
            "bid": raw_data.get("HIGHDAY", None),
            "ask": raw_data.get("LOWDAY", None),
            "volume_24h": raw_data.get("VOLUME24HOUR", 0),
            "volume_24h_to": raw_data.get("VOLUME24HOURTO", 0),
            "percent_change_24h": raw_data.get("CHANGEPCT24HOUR", 0),
            "market_cap": raw_data.get("MKTCAP", 0),
            "high_24h": raw_data.get("HIGHDAY", 0),
            "low_24h": raw_data.get("LOWDAY", 0),
        }
        
        return result
    
    except Exception as e:
        raise APIError(f"CryptoCompare fetch market data failed for {symbol}: {str(e)}")

@async_exception_handler
async def fetch_price(api_manager: Any, symbol: str) -> float:
    """
    Fetch current price for a symbol using CryptoCompare.
    
    Args:
        api_manager: API Manager instance
        symbol (str): Asset symbol
        
    Returns:
        float: Current price
        
    Raises:
        APIError: If the request fails
    """
    try:
        # Get API key from config if available
        api_key = api_manager.api_keys.get("cryptocompare", "")
        headers = {"authorization": f"Apikey {api_key}"} if api_key else None
        
        # Get current price data
        price_url = f"{CRYPTOCOMPARE_API_URL}/price"
        price_params = {
            "fsym": symbol.upper(),
            "tsyms": "USD",
            "e": "CCCAGG"
        }
        
        price_data = await api_manager.make_request(
            api_name="cryptocompare",
            method="GET",
            url=price_url,
            params=price_params,
            headers=headers,
            cache_expiry=30  # Cache for 30 seconds
        )
        
        # Extract price
        price = price_data.get("USD")
        
        if price is None:
            raise APIError(f"CryptoCompare price data not found for {symbol}")
        
        return float(price)
    
    except Exception as e:
        raise APIError(f"CryptoCompare fetch price failed for {symbol}: {str(e)}")

@async_exception_handler
async def get_available_assets(api_manager: Any) -> List[str]:
    """
    Get list of available assets using CryptoCompare.
    
    Args:
        api_manager: API Manager instance
        
    Returns:
        List[str]: List of available asset symbols
        
    Raises:
        APIError: If the request fails
    """
    try:
        # Get API key from config if available
        api_key = api_manager.api_keys.get("cryptocompare", "")
        headers = {"authorization": f"Apikey {api_key}"} if api_key else None
        
        # Get top coins by market cap
        top_coins_url = f"{CRYPTOCOMPARE_API_URL}/top/mktcapfull"
        params = {
            "limit": 100,
            "tsym": "USD"
        }
        
        top_coins_data = await api_manager.make_request(
            api_name="cryptocompare",
            method="GET",
            url=top_coins_url,
            params=params,
            headers=headers,
            cache_expiry=300  # Cache for 5 minutes
        )
        
        # Extract symbols
        if "Data" not in top_coins_data:
            raise APIError("CryptoCompare top coins data not found")
        
        symbols = [
            coin.get("CoinInfo", {}).get("Name", "").upper()
            for coin in top_coins_data["Data"]
        ]
        
        # Remove duplicates and sort
        symbols = sorted(list(set(symbols)))
        
        return symbols
    
    except Exception as e:
        raise APIError(f"CryptoCompare get available assets failed: {str(e)}")