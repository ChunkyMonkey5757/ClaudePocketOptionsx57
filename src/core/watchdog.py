
"""
Watchdog Monitor for PocketBotX57.
Protects critical async signal loops from freezing or crashing by restarting them.
"""

import asyncio
import time
from typing import Callable, Dict
from datetime import datetime, timedelta
from src.utils.logger import get_logger

logger = get_logger("watchdog")

class Watchdog:
    """
    Tracks last heartbeat of monitored processes.
    If inactive too long, triggers recovery handler.
    """

    def __init__(self, timeout_seconds: int = 120):
        self.timeout = timedelta(seconds=timeout_seconds)
        self.heartbeats: Dict[str, datetime] = {}
        self.recovery_handlers: Dict[str, Callable] = {}

    def register_task(self, name: str, recovery_function: Callable):
        """
        Register a named task with a recovery handler.

        Args:
            name: Unique identifier
            recovery_function: Async function to restart or reset task
        """
        self.heartbeats[name] = datetime.utcnow()
        self.recovery_handlers[name] = recovery_function
        logger.info(f"Watchdog registered for task: {name}")

    def beat(self, name: str):
        """
        Manually update the heartbeat for a task.

        Args:
            name: Identifier of task
        """
        self.heartbeats[name] = datetime.utcnow()

    async def monitor(self, interval: int = 10):
        """
        Continuous loop that checks for frozen or stalled tasks.

        Args:
            interval: Time between watchdog checks (seconds)
        """
        while True:
            now = datetime.utcnow()
            for name, last_beat in self.heartbeats.items():
                if now - last_beat > self.timeout:
                    logger.warning(f"Watchdog triggered for task: {name}")
                    if name in self.recovery_handlers:
                        try:
                            await self.recovery_handlers[name]()
                            self.heartbeats[name] = datetime.utcnow()
                            logger.info(f"Task {name} restarted by watchdog.")
                        except Exception as e:
                            logger.error(f"Watchdog recovery failed for {name}: {str(e)}")
            await asyncio.sleep(interval)
