
"""
Utility Toolkit for PocketBotX57.
Provides general-purpose formatting, math helpers, time utilities, and common tools.
"""

import datetime
import pytz
from typing import Union

def now_utc_iso() -> str:
    """
    Returns current UTC time as ISO string.
    """
    return datetime.datetime.utcnow().isoformat()

def format_timestamp(ts: Union[int, float]) -> str:
    """
    Converts a UNIX timestamp into human-readable format.

    Args:
        ts: Timestamp in seconds

    Returns:
        str: "YYYY-MM-DD HH:MM:SS"
    """
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def round_float(value: float, digits: int = 2) -> float:
    """
    Rounds a float to X digits.

    Args:
        value: Number to round
        digits: Number of decimal places

    Returns:
        float
    """
    return round(value, digits)

def to_percent(value: float, max_value: float = 1.0, digits: int = 2) -> float:
    """
    Converts decimal to percent (0.75 → 75%).

    Args:
        value: Decimal value
        max_value: Max base (default=1)
        digits: Rounding

    Returns:
        float: Percent
    """
    return round((value / max_value) * 100, digits)

def is_between(val: float, low: float, high: float) -> bool:
    """
    Checks if value is between low and high.

    Args:
        val: Number
        low: Lower bound
        high: Upper bound

    Returns:
        bool
    """
    return low <= val <= high

def timestamp_now() -> int:
    """
    Returns current UTC time as UNIX timestamp.
    """
    return int(datetime.datetime.utcnow().timestamp())
