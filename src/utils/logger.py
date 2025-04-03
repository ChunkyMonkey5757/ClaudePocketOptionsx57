
"""
Smart Logging Utility for PocketBotX57.
Supports rich console logging, file output (optional), level filtering, and structured messages.
"""

import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Creates or returns a pre-configured logger.

    Args:
        name: Name of logger
        level: Logging level

    Returns:
        Logger object
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Prevent duplicate handlers

    logger.setLevel(level)

    # Console Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Optional - Uncomment to enable persistent logs)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_file = os.path.join(LOG_DIR, f"{name}_{timestamp}.log")
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    return logger
