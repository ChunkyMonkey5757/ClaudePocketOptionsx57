
"""
CoinMarketCap Adapter for PocketBotX57.
Fetches live price and metadata using the official CMC REST API.
"""

import os
import requests
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("coinmarketcap")

class CoinMarketCapAdapter:
    BASE_URL = "https://pro-api.coinmarketcap.com/v1"

    def __init__(self):
        self.api_key = os.getenv("CMC_API_KEY")
        if not self.api_key:
            logger.warning("CoinMarketCap API key not found in environment variables.")

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Fetch live price of a symbol from CoinMarketCap.

        Args:
            symbol: e.g. "BTC", "ETH"

        Returns:
            Price as float or None
        """
        try:
            endpoint = f"{self.BASE_URL}/cryptocurrency/quotes/latest"
            headers = {"X-CMC_PRO_API_KEY": self.api_key}
            params = {"symbol": symbol.upper(), "convert": "USD"}

            response = requests.get(endpoint, headers=headers, params=params, timeout=10)
            data = response.json()

            price = data["data"][symbol.upper()]["quote"]["USD"]["price"]
            return float(price)

        except Exception as e:
            logger.error(f"CMC price fetch failed for {symbol}: {e}")
            return None

    def get_metadata(self, symbol: str) -> Optional[dict]:
        """
        Fetch metadata (description, rank, supply, etc.) for a given symbol.

        Args:
            symbol: e.g. "BTC"

        Returns:
            Dict of metadata or None
        """
        try:
            endpoint = f"{self.BASE_URL}/cryptocurrency/info"
            headers = {"X-CMC_PRO_API_KEY": self.api_key}
            params = {"symbol": symbol.upper()}

            response = requests.get(endpoint, headers=headers, params=params, timeout=10)
            data = response.json()
            return data["data"][symbol.upper()]

        except Exception as e:
            logger.error(f"CMC metadata fetch failed for {symbol}: {e}")
            return None
