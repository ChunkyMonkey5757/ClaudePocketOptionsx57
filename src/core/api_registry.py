
"""
Advanced Multi-Source API Registry for PocketBotX57.
Routes OHLC, price, forecast, and metadata requests using verified adapters only.
"""

from typing import Dict, List, Optional, Any
from src.api_handler.coingecko_adapter import CoinGeckoAdapter
from src.api_handler.cryptocompare_adapter import CryptoCompareAdapter
from src.utils.logger import get_logger

logger = get_logger("api_registry")

class APIRegistry:
    """
    Smart router for data providers with fallback and logging.
    Only uses verified, uploaded adapters — no placeholders.
    """

    def __init__(self):
        self.providers = {
            "coingecko": CoinGeckoAdapter(),
            "cryptocompare": CryptoCompareAdapter()
        }

        # Define fallback priority chains for supported data types
        self.ohlc_priority = ["coingecko", "cryptocompare"]
        self.price_priority = ["coingecko", "cryptocompare"]

    def _attempt_chain(self, symbol: str, method: str, chain: List[str], **kwargs) -> Optional[Any]:
        """
        Attempts a method across multiple providers in a chain until one succeeds.

        Args:
            symbol: Asset symbol
            method: Method to call (get_price, get_ohlc, etc.)
            chain: Provider names in fallback order

        Returns:
            Successful result or None
        """
        for name in chain:
            provider = self.providers.get(name)
            try:
                if not provider or not hasattr(provider, method):
                    continue
                fn = getattr(provider, method)
                result = fn(symbol, **kwargs)
                if result:
                    logger.info(f"[{method.upper()}] {name} succeeded for {symbol}")
                    return result
            except Exception as e:
                logger.warning(f"[{method.upper()}] {name} failed for {symbol}: {e}")
                continue

        logger.error(f"[{method.upper()}] All providers failed for {symbol}")
        return None

    def fetch_price(self, symbol: str) -> Optional[float]:
        return self._attempt_chain(symbol, "get_price", self.price_priority)

    def fetch_ohlc(self, symbol: str, interval: str = "1m", limit: int = 50) -> Optional[List[Dict]]:
        return self._attempt_chain(symbol, "get_ohlc", self.ohlc_priority, interval=interval, limit=limit)
