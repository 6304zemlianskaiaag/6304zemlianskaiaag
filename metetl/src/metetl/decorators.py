import functools
import time
import os
from .logging_config import get_logger

logger = get_logger(__name__)


def time_method(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.debug(f"{func.__name__} (PID: {os.getpid()}): {end - start:.2f} сек.")
        return result
    return wrapper


def time_method_async(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        logger.debug(f"[АСИНХРОН] {func.__name__} (PID: {os.getpid()}): {end - start:.2f} сек.")
        return result
    return wrapper