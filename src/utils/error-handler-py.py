"""
Error handling utility for PocketBotX57.
Provides error tracking, reporting, and recovery mechanisms.
"""

import sys
import traceback
import functools
import asyncio
import logging
from typing import Any, Callable, TypeVar, Optional, Type, cast, Dict, List, Union

from src.utils.logger import get_logger

# Type variables for function annotations
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])

# Get logger
logger = get_logger("error_handler")

class BotError(Exception):
    """Base exception class for PocketBotX57 errors."""
    def __init__(self, message: str, error_code: Optional[str] = None) -> None:
        self.message = message
        self.error_code = error_code
        super().__init__(message)

class APIError(BotError):
    """Exception raised for API-related errors."""
    pass

class DataError(BotError):
    """Exception raised for data-related errors."""
    pass

class ConfigError(BotError):
    """Exception raised for configuration-related errors."""
    pass

class SignalError(BotError):
    """Exception raised for signal generation errors."""
    pass

class TelegramError(BotError):
    """Exception raised for Telegram-related errors."""
    pass

def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Central error handler function.
    
    Args:
        error (Exception): The exception that occurred
        context (Dict[str, Any], optional): Additional context information. Defaults to None.
    """
    error_context = context or {}
    
    # Get traceback information
    tb = traceback.format_exception(type(error), error, error.__traceback__)
    tb_str = ''.join(tb)
    
    # Log the error with context
    logger.error(
        f"Unhandled error: {type(error).__name__}: {str(error)}\n"
        f"Context: {error_context}\n"
        f"Traceback:\n{tb_str}"
    )
    
    # Additional error handling logic could be added here:
    # - Send error notifications
    # - Attempt recovery
    # - etc.

def exception_handler(func: F) -> F:
    """
    Decorator for handling exceptions in functions.
    
    Args:
        func (F): The function to decorate
        
    Returns:
        F: Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            handle_error(e, {"function": func.__name__, "args": args, "kwargs": kwargs})
            raise
    
    return cast(F, wrapper)

def async_exception_handler(func: AsyncF) -> AsyncF:
    """
    Decorator for handling exceptions in async functions.
    
    Args:
        func (AsyncF): The async function to decorate
        
    Returns:
        AsyncF: Decorated async function
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            handle_error(e, {"function": func.__name__, "args": args, "kwargs": kwargs})
            raise
    
    return cast(AsyncF, wrapper)

def setup_global_exception_handlers() -> None:
    """Set up global exception handlers for unhandled exceptions."""
    
    def global_exception_handler(exc_type: Type[BaseException], 
                               exc_value: BaseException, 
                               exc_traceback: traceback.TracebackType) -> None:
        """Global exception handler for unhandled exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't intercept keyboard interrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    def global_async_exception_handler(loop: asyncio.AbstractEventLoop, 
                                     context: Dict[str, Any]) -> None:
        """Global exception handler for unhandled async exceptions."""
        exception = context.get("exception")
        if exception:
            handle_error(exception, context)
        else:
            logger.error(f"Unhandled async error: {context.get('message')}\nContext: {context}")
    
    # Set the global exception handlers
    sys.excepthook = global_exception_handler
    
    # Set the async exception handler
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(global_async_exception_handler)
    
    logger.info("Global exception handlers have been set up")

def error_to_user_message(error: Exception) -> str:
    """
    Convert an error to a user-friendly message.
    
    Args:
        error (Exception): The exception
        
    Returns:
        str: User-friendly error message
    """
    if isinstance(error, APIError):
        return f"⚠️ API Error: {error.message}"
    elif isinstance(error, DataError):
        return f"⚠️ Data Error: {error.message}"
    elif isinstance(error, ConfigError):
        return f"⚠️ Configuration Error: {error.message}"
    elif isinstance(error, SignalError):
        return f"⚠️ Signal Error: {error.message}"
    elif isinstance(error, TelegramError):
        return f"⚠️ Telegram Error: {error.message}"
    else:
        return f"⚠️ An error occurred: {str(error)}"