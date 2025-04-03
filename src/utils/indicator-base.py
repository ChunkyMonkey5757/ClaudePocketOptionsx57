"""
Base Indicator class for PocketBotX57.
Defines the interface that all technical indicators must implement.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import logging

class IndicatorBase(ABC):
    """
    Abstract base class for all technical indicators.
    All indicator strategies must inherit from this class.
    """
    
    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the indicator with parameters.
        
        Args:
            params: Dictionary with indicator parameters
        """
        self.params = params
        self.name = params.get("name", "base_indicator")
        self.enabled = params.get("enabled", True)
        self.weight = params.get("weight", 1.0)
    
    @abstractmethod
    async def calculate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicator values from market data.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dictionary with indicator values and analysis
        """
        pass
    
    @abstractmethod
    async def get_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal based on indicator analysis.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Signal dictionary with direction and confidence
        """
        pass
    
    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update indicator parameters.
        
        Args:
            new_params: Dictionary with parameters to update
        """
        for key, value in new_params.items():
            if key in self.params:
                self.params[key] = value
            
            # Special case for nested signal_strength dictionary
            if key == "signal_strength" and isinstance(value, dict):
                if "signal_strength" not in self.params:
                    self.params["signal_strength"] = {}
                    
                for subkey, subvalue in value.items():
                    self.params["signal_strength"][subkey] = subvalue