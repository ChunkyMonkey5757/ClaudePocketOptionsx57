
"""
Alpha Vantage Adapter for PocketBotX57.
Fetches technical indicators and OHLC data using Alpha Vantage REST API.
"""

import os
import requests
from typing import Optional, List, Dict
from src.utils.logger import get_logger

logger = get_logger("alphavantage")

class AlphaVantageAdapter:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            logger.warning("Alpha Vantage API key not found in environment variables.")

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Fetch latest price using GLOBAL_QUOTE.

        Args:
            symbol: e.g., "BTC/USD"

        Returns:
            Price as float or None
        """
        try:
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": symbol.upper().split("/")[0],
                "to_currency": symbol.upper().split("/")[1] if "/" in symbol else "USD",
                "apikey": self.api_key
            }
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            data = response.json()
            price = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
            return float(price)
        except Exception as e:
            logger.error(f"Alpha Vantage price fetch failed for {symbol}: {e}")
            return None

    def get_ohlc(self, symbol: str, interval: str = "1min", limit: int = 50) -> Optional[List[Dict]]:
        """
        Fetch OHLC candles using TIME_SERIES_CRYPTO_INTRADAY.

        Args:
            symbol: e.g., "BTC"
            interval: "1min", "5min", etc.
            limit: number of bars

        Returns:
            List of OHLC dicts
        """
        try:
            params = {
                "function": "CRYPTO_INTRADAY",
                "symbol": symbol.upper(),
                "market": "USD",
                "interval": interval,
                "apikey": self.api_key
            }
            response = requests.get(self.BASE_URL, params=params, timeout=15)
            data = response.json()
            key = f"Time Series Crypto ({interval})"
            series = data.get(key, {})

            candles = []
            for timestamp, values in list(series.items())[:limit]:
                candles.append({
                    "timestamp": timestamp,
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": float(values["5. volume"])
                })

            return candles[::-1]  # Return in ascending order

        except Exception as e:
            logger.error(f"Alpha Vantage OHLC fetch failed for {symbol}: {e}")
            return None
