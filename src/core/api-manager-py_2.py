"""
API Manager module for PocketBotX57.
Manages multiple API connections with fallback mechanisms.
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import aiohttp

from src.config import get_api_keys, API_RETRY_COUNT, API_TIMEOUT_SECONDS
from src.utils.logger import get_logger
from src.utils.error_handler import APIError, async_exception_handler

# Get logger
logger = get_logger("api_manager")

class ApiManager:
    """
    API Manager class to handle multiple data sources with fallback.
    Provides unified interface to fetch market data.
    """
    
    def __init__(self):
        """Initialize the API Manager."""
        self.api_keys = get_api_keys()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # API request tracking
        self.request_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_reset = time.time()
        self.reset_interval = 3600  # Reset counts every hour
        
        # API priority order
        self.api_priority = ["ccxt", "coingecko", "cryptocompare", "alpha_vantage"]
        
        # Cache mechanism
        self.cache: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, datetime] = {}
    
    async def initialize(self) -> None:
        """Initialize the API client session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS)
            )
            logger.info("API Manager session initialized")
    
    async def close(self) -> None:
        """Close the API client session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("API Manager session closed")
    
    def _reset_counters(self) -> None:
        """Reset API request and error counters if reset interval has passed."""
        current_time = time.time()
        if current_time - self.last_reset > self.reset_interval:
            self.request_counts = {}
            self.error_counts = {}
            self.last_reset = current_time
            logger.debug("API request and error counters reset")
    
    def _increment_request_count(self, api_name: str) -> None:
        """Increment the request count for an API."""
        self.request_counts[api_name] = self.request_counts.get(api_name, 0) + 1
    
    def _increment_error_count(self, api_name: str) -> None:
        """Increment the error count for an API."""
        self.error_counts[api_name] = self.error_counts.get(api_name, 0) + 1
    
    def _get_cache_key(self, api_name: str, endpoint: str, params: Dict[str, Any]) -> str:
        """
        Generate a cache key from API name, endpoint, and parameters.
        
        Args:
            api_name (str): Name of the API
            endpoint (str): API endpoint
            params (Dict[str, Any]): Request parameters
            
        Returns:
            str: Cache key
        """
        param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        return f"{api_name}:{endpoint}:{param_str}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            cache_key (str): Cache key
            
        Returns:
            bool: True if cache is valid, False otherwise
        """
        return (
            cache_key in self.cache and
            cache_key in self.cache_expiry and
            datetime.now() < self.cache_expiry[cache_key]
        )
    
    def _store_in_cache(self, cache_key: str, data: Any, expiry_seconds: int = 60) -> None:
        """
        Store data in cache with expiration.
        
        Args:
            cache_key (str): Cache key
            data (Any): Data to cache
            expiry_seconds (int, optional): Cache expiry in seconds. Defaults to 60.
        """
        self.cache[cache_key] = data
        self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=expiry_seconds)
    
    @async_exception_handler
    async def make_request(self, api_name: str, method: str, url: str, 
                         params: Optional[Dict[str, Any]] = None, 
                         headers: Optional[Dict[str, Any]] = None,
                         data: Optional[Dict[str, Any]] = None,
                         cache_expiry: int = 60) -> Dict[str, Any]:
        """
        Make an API request with caching and retry logic.
        
        Args:
            api_name (str): Name of the API service
            method (str): HTTP method (GET, POST, etc.)
            url (str): Request URL
            params (Dict[str, Any], optional): Query parameters. Defaults to None.
            headers (Dict[str, Any], optional): HTTP headers. Defaults to None.
            data (Dict[str, Any], optional): Request body data. Defaults to None.
            cache_expiry (int, optional): Cache expiry in seconds. Defaults to 60.
            
        Returns:
            Dict[str, Any]: API response
            
        Raises:
            APIError: If the request fails after retries
        """
        params = params or {}
        headers = headers or {}
        
        # Reset counters if needed
        self._reset_counters()
        
        # Check cache first
        cache_key = self._get_cache_key(api_name, url, params)
        if self._is_cache_valid(cache_key):
            logger.debug(f"Using cached data for {api_name}: {url}")
            return self.cache[cache_key]
        
        # Initialize session if needed
        await self.initialize()
        
        # Track API request
        self._increment_request_count(api_name)
        
        # Retry logic
        retries = 0
        while retries < API_RETRY_COUNT:
            try:
                if method.upper() == "GET":
                    async with self.session.get(url, params=params, headers=headers) as response:
                        if response.status in (200, 201, 202):
                            result = await response.json()
                            
                            # Cache successful result
                            self._store_in_cache(cache_key, result, cache_expiry)
                            
                            return result
                        else:
                            error_text = await response.text()
                            logger.warning(
                                f"API request failed: {api_name}, Status: {response.status}, Error: {error_text}"
                            )
                
                elif method.upper() == "POST":
                    async with self.session.post(url, params=params, json=data, headers=headers) as response:
                        if response.status in (200, 201, 202):
                            result = await response.json()
                            
                            # Cache successful result if it's a read operation
                            if not data:  # Only cache GET-like POST requests without body
                                self._store_in_cache(cache_key, result, cache_expiry)
                            
                            return result
                        else:
                            error_text = await response.text()
                            logger.warning(
                                f"API request failed: {api_name}, Status: {response.status}, Error: {error_text}"
                            )
                
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
            
            except Exception as e:
                logger.warning(f"API request error: {api_name}, Attempt {retries + 1}/{API_RETRY_COUNT}: {str(e)}")
            
            # Increment retry count and wait before retrying
            retries += 1
            if retries < API_RETRY_COUNT:
                # Exponential backoff
                await asyncio.sleep(2 ** retries)
        
        # All retries failed
        self._increment_error_count(api_name)
        raise APIError(f"API request failed after {API_RETRY_COUNT} retries: {api_name} {url}")
    
    async def fetch_market_data(self, symbol: str, interval: str = "1m", 
                              limit: int = 100) -> Dict[str, Any]:
        """
        Fetch market data with automatic fallback between APIs.
        
        Args:
            symbol (str): Asset symbol (e.g., BTC, ETH)
            interval (str, optional): Time interval. Defaults to "1m".
            limit (int, optional): Number of data points. Defaults to 100.
            
        Returns:
            Dict[str, Any]: Market data
            
        Raises:
            APIError: If all API sources fail
        """
        # Try APIs in priority order
        for api_name in self.api_priority:
            try:
                if api_name == "ccxt":
                    from src.api_handler.ccxt_adapter import fetch_market_data as fetch_ccxt
                    return await fetch_ccxt(self, symbol, interval, limit)
                    
                elif api_name == "coingecko":
                    from src.api_handler.coingecko_adapter import fetch_market_data as fetch_coingecko
                    return await fetch_coingecko(self, symbol, interval, limit)
                    
                elif api_name == "cryptocompare":
                    from src.api_handler.cryptocompare_adapter import fetch_market_data as fetch_cryptocompare
                    return await fetch_cryptocompare(self, symbol, interval, limit)
                    
                elif api_name == "alpha_vantage":
                    from src.api_handler.alpha_vantage_adapter import fetch_market_data as fetch_alpha_vantage
                    return await fetch_alpha_vantage(self, symbol, interval, limit)
                
            except Exception as e:
                logger.warning(f"API source {api_name} failed for {symbol}: {str(e)}")
                continue
        
        # All APIs failed
        raise APIError(f"All API sources failed for {symbol}")
    
    async def fetch_price(self, symbol: str) -> float:
        """
        Fetch current price for a symbol with fallback.
        
        Args:
            symbol (str): Asset symbol
            
        Returns:
            float: Current price
            
        Raises:
            APIError: If all API sources fail
        """
        # Try APIs in priority order
        for api_name in self.api_priority:
            try:
                if api_name == "ccxt":
                    from src.api_handler.ccxt_adapter import fetch_price as fetch_ccxt_price
                    return await fetch_ccxt_price(self, symbol)
                    
                elif api_name == "coingecko":
                    from src.api_handler.coingecko_adapter import fetch_price as fetch_coingecko_price
                    return await fetch_coingecko_price(self, symbol)
                    
                elif api_name == "cryptocompare":
                    from src.api_handler.cryptocompare_adapter import fetch_price as fetch_cryptocompare_price
                    return await fetch_cryptocompare_price(self, symbol)
                    
                elif api_name == "alpha_vantage":
                    from src.api_handler.alpha_vantage_adapter import fetch_price as fetch_alpha_vantage_price
                    return await fetch_alpha_vantage_price(self, symbol)
                
            except Exception as e:
                logger.warning(f"API source {api_name} failed to fetch price for {symbol}: {str(e)}")
                continue
        
        # All APIs failed
        raise APIError(f"All API sources failed to fetch price for {symbol}")
    
    async def fetch_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch sentiment data for a symbol.
        
        Args:
            symbol (str): Asset symbol
            
        Returns:
            Dict[str, Any]: Sentiment data
        """
        # This is a placeholder that will be expanded with actual sentiment data sources
        try:
            # Try to fetch from cryptocurrency news aggregators or sentiment APIs
            # For now, return a placeholder
            return {
                "symbol": symbol,
                "sentiment": "neutral",
                "score": 0.5,
                "sources": []
            }
        except Exception as e:
            logger.warning(f"Failed to fetch sentiment data for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "sentiment": "neutral",
                "score": 0.5,
                "sources": [],
                "error": str(e)
            }
    
    async def get_available_assets(self) -> List[str]:
        """
        Get list of available assets from the API.
        
        Returns:
            List[str]: List of available asset symbols
        """
        try:
            # Try to fetch from CCXT first
            from src.api_handler.ccxt_adapter import get_available_assets as get_ccxt_assets
            return await get_ccxt_assets(self)
        except Exception as e:
            logger.warning(f"Failed to get available assets from CCXT: {str(e)}")
            
            # Fallback to default assets from config
            from src.config import SUPPORTED_ASSETS
            return SUPPORTED_ASSETS