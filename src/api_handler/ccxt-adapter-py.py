"""
CCXT Adapter for PocketBotX57.
Provides cryptocurrency market data using the CCXT library.
"""

import asyncio
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import ccxt.async_support as ccxt

from src.utils.logger import get_logger
from src.utils.error_handler import APIError, async_exception_handler

# Get logger
logger = get_logger("ccxt_adapter")

# Mapping of common intervals to CCXT timeframes
TIMEFRAME_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1d",
}

# Mapping of symbols to CCXT market symbols
SYMBOL_MAP = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "BNB": "BNB/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "ADA": "ADA/USDT",
    "DOGE": "DOGE/USDT",
    "MATIC": "MATIC/USDT",
    "DOT": "DOT/USDT",
    "AVAX": "AVAX/USDT",
    "LINK": "LINK/USDT",
    "UNI": "UNI/USDT",
    "ATOM": "ATOM/USDT",
}

# List of exchanges to try, in priority order
EXCHANGES = ["binance", "kucoin", "okx", "bybit"]

@async_exception_handler
async def get_exchange(api_manager: Any) -> ccxt.Exchange:
    """
    Get a CCXT exchange instance.
    
    Args:
        api_manager: API Manager instance
        
    Returns:
        ccxt.Exchange: CCXT exchange instance
        
    Raises:
        APIError: If no exchanges are available
    """
    for exchange_id in EXCHANGES:
        try:
            # Create exchange instance
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000,
                'apiKey': api_manager.api_keys.get(f"{exchange_id}_api_key", ""),
                'secret': api_manager.api_keys.get(f"{exchange_id}_api_secret", ""),
            })
            
            # Test connection
            await exchange.load_markets()
            logger.info(f"Using CCXT exchange: {exchange_id}")
            return exchange
        except Exception as e:
            logger.warning(f"Failed to initialize CCXT exchange {exchange_id}: {str(e)}")
            continue
    
    raise APIError("No CCXT exchanges available")

@async_exception_handler
async def fetch_market_data(api_manager: Any, symbol: str, interval: str = "1m", 
                          limit: int = 100) -> Dict[str, Any]:
    """
    Fetch market data using CCXT.
    
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
    # Get exchange
    exchange = await get_exchange(api_manager)
    
    try:
        # Get appropriate market symbol
        market_symbol = SYMBOL_MAP.get(symbol.upper(), f"{symbol.upper()}/USDT")
        
        # Get appropriate timeframe
        timeframe = TIMEFRAME_MAP.get(interval, "1m")
        
        # Fetch OHLCV data
        ohlcv = await exchange.fetch_ohlcv(
            symbol=market_symbol,
            timeframe=timeframe,
            limit=limit
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Get ticker data for additional information
        ticker = await exchange.fetch_ticker(market_symbol)
        
        # Construct result
        result = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "exchange": exchange.id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "ohlc": df.to_dict('records'),
            "price": ticker['last'],
            "bid": ticker['bid'],
            "ask": ticker['ask'],
            "volume_24h": ticker['quoteVolume'],
            "percent_change_24h": ticker['percentage'],
        }
        
        # Clean up
        await exchange.close()
        
        return result
    
    except Exception as e:
        # Clean up
        await exchange.close()
        
        # Re-raise error
        raise APIError(f"CCXT fetch market data failed for {symbol}: {str(e)}")

@async_exception_handler
async def fetch_price(api_manager: Any, symbol: str) -> float:
    """
    Fetch current price for a symbol using CCXT.
    
    Args:
        api_manager: API Manager instance
        symbol (str): Asset symbol
        
    Returns:
        float: Current price
        
    Raises:
        APIError: If the request fails
    """
    # Get exchange
    exchange = await get_exchange(api_manager)
    
    try:
        # Get appropriate market symbol
        market_symbol = SYMBOL_MAP.get(symbol.upper(), f"{symbol.upper()}/USDT")
        
        # Fetch ticker
        ticker = await exchange.fetch_ticker(market_symbol)
        
        # Get price
        price = ticker['last']
        
        # Clean up
        await exchange.close()
        
        return price
    
    except Exception as e:
        # Clean up
        await exchange.close()
        
        # Re-raise error
        raise APIError(f"CCXT fetch price failed for {symbol}: {str(e)}")

@async_exception_handler
async def get_available_assets(api_manager: Any) -> List[str]:
    """
    Get list of available assets using CCXT.
    
    Args:
        api_manager: API Manager instance
        
    Returns:
        List[str]: List of available asset symbols
        
    Raises:
        APIError: If the request fails
    """
    # Get exchange
    exchange = await get_exchange(api_manager)
    
    try:
        # Fetch markets
        markets = await exchange.fetch_markets()
        
        # Extract symbols (base currencies with USDT quote)
        symbols = [market['base'] for market in markets if market['quote'] == 'USDT']
        
        # Remove duplicates and sort
        symbols = sorted(list(set(symbols)))
        
        # Clean up
        await exchange.close()
        
        return symbols
    
    except Exception as e:
        # Clean up
        await exchange.close()
        
        # Re-raise error
        raise APIError(f"CCXT get available assets failed: {str(e)}")