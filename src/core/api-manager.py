"""
API Manager for PocketBotX57.
Manages multiple API connections with fallback mechanisms.
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import aiohttp
import logging

from src.utils.logger import get_logger
from src.utils.error_handler import APIError

logger = get_logger(__name__)

class APIManager:
    """
    API Manager class to handle multiple data sources with fallback.
    Provides unified interface to fetch market data.
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        """
        Initialize the API Manager with API keys.
        
        Args:
            api_keys: Dictionary with API keys
        """
        self.api_keys = api_keys
        self.session = None
        
        # API rate limiting
        self.rate_limits = {
            "alpha_vantage": {"calls": 0, "last_reset": time.time(), "max_calls": 5, "reset_interval": 60},
            "cryptocompare": {"calls": 0, "last_reset": time.time(), "max_calls": 30, "reset_interval": 60},
            "coingecko": {"calls": 0, "last_reset": time.time(), "max_calls": 10, "reset_interval": 60},
            "coinmarketcap": {"calls": 0, "last_reset": time.time(), "max_calls": 10, "reset_interval": 60},
        }
        
        # API priority (ordered by reliability and data quality)
        self.api_priority = ["alpha_vantage", "cryptocompare", "coingecko", "coinmarketcap"]
        
        # Cache mechanism
        self.cache = {}
        self.cache_expiry = {}
        
        # Error tracking
        self.error_counts = {}
        
        logger.info("API Manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the API client session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            logger.info("API session initialized")
    
    async def close(self) -> None:
        """Close the API client session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("API session closed")
    
    def _update_rate_limit(self, api_name: str) -> bool:
        """
        Update rate limit counters and return True if request can proceed.
        
        Args:
            api_name: Name of the API
            
        Returns:
            Boolean indicating if request can proceed
        """
        if api_name not in self.rate_limits:
            return True
            
        now = time.time()
        limit_info = self.rate_limits[api_name]
        
        # Reset counter if interval has passed
        if now - limit_info["last_reset"] > limit_info["reset_interval"]:
            limit_info["calls"] = 0
            limit_info["last_reset"] = now
        
        # Check if we're under the limit
        if limit_info["calls"] < limit_info["max_calls"]:
            limit_info["calls"] += 1
            return True
        
        return False
    
    def _get_cache_key(self, api_name: str, endpoint: str, params: Dict[str, Any]) -> str:
        """
        Generate cache key from API endpoint and parameters.
        
        Args:
            api_name: Name of the API
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Cache key string
        """
        param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        return f"{api_name}:{endpoint}:{param_str}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve data from cache if valid.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached data or None if expired/not found
        """
        if cache_key in self.cache and cache_key in self.cache_expiry:
            if datetime.now() < self.cache_expiry[cache_key]:
                return self.cache[cache_key]
        return None
    
    def _store_in_cache(self, cache_key: str, data: Any, expiry_seconds: int = 60) -> None:
        """
        Store data in cache with expiration.
        
        Args:
            cache_key: Cache key
            data: Data to cache
            expiry_seconds: Cache expiration in seconds
        """
        self.cache[cache_key] = data
        self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=expiry_seconds)
    
    async def make_request(self, api_name: str, method: str, url: str, 
                         params: Optional[Dict[str, Any]] = None, 
                         headers: Optional[Dict[str, str]] = None,
                         data: Optional[Dict[str, Any]] = None,
                         cache_expiry: int = 60) -> Dict[str, Any]:
        """
        Make an API request with caching and retry logic.
        
        Args:
            api_name: Name of the API
            method: HTTP method (GET, POST)
            url: Request URL
            params: Query parameters
            headers: HTTP headers
            data: Request body data
            cache_expiry: Cache expiration in seconds
            
        Returns:
            API response
            
        Raises:
            APIError: If the request fails
        """
        params = params or {}
        headers = headers or {}
        
        # Check rate limits
        if not self._update_rate_limit(api_name):
            logger.warning("Rate limit exceeded for %s", api_name)
            raise APIError(f"Rate limit exceeded for {api_name}")
        
        # Check cache
        cache_key = self._get_cache_key(api_name, url, params)
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            logger.debug("Cache hit for %s: %s", api_name, url)
            return cached_data
        
        # Initialize session if needed
        await self.initialize()
        
        # Make request with retry logic
        max_retries = 3
        retry_delay = 1
        
        for retry in range(max_retries):
            try:
                if method.upper() == "GET":
                    async with self.session.get(url, params=params, headers=headers, timeout=10) as response:
                        if response.status == 200:
                            result = await response.json()
                            self._store_in_cache(cache_key, result, cache_expiry)
                            return result
                        else:
                            text = await response.text()
                            logger.warning("API error (%s): %s - %s", api_name, response.status, text)
                            
                elif method.upper() == "POST":
                    async with self.session.post(url, params=params, json=data, headers=headers, timeout=10) as response:
                        if response.status == 200:
                            result = await response.json()
                            self._store_in_cache(cache_key, result, cache_expiry)
                            return result
                        else:
                            text = await response.text()
                            logger.warning("API error (%s): %s - %s", api_name, response.status, text)
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning("Request failed (%s): %s. Retry %d/%d", api_name, str(e), retry + 1, max_retries)
                
                if retry < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** retry))  # Exponential backoff
                    continue
            
            # Increment error count for this API
            self.error_counts[api_name] = self.error_counts.get(api_name, 0) + 1
        
        raise APIError(f"Request failed after {max_retries} retries: {api_name} - {url}")
    
    async def fetch_market_data(self, symbol: str, interval: str = "1m", limit: int = 100) -> Dict[str, Any]:
        """
        Fetch market data with fallback between multiple data sources.
        
        Args:
            symbol: Asset symbol (e.g., BTC, ETH)
            interval: Time interval (e.g., 1m, 5m, 15m, 1h)
            limit: Number of data points
            
        Returns:
            Market data dictionary
            
        Raises:
            APIError: If all data sources fail
        """
        # Try each API in priority order
        errors = []
        
        for api_name in self.api_priority:
            try:
                if api_name == "alpha_vantage":
                    from src.api_handler.alpha_vantage import fetch_market_data
                    return await fetch_market_data(self, symbol, interval, limit)
                
                elif api_name == "cryptocompare":
                    from src.api_handler.cryptocompare import fetch_market_data