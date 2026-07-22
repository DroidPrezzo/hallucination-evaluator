from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

ErrorClassifier = Callable[[Exception], Tuple[str, Optional[float]]]


@dataclass
class RetryConfig:
    max_attempts_server: int = 6
    max_attempts_timeout: int = 2
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0
    jitter_max_s: float = 1.0


def retry_api_call(
    fn: Callable[[], T],
    classify: ErrorClassifier,
    config: Optional[RetryConfig] = None,
) -> T:
    if config is None:
        config = RetryConfig()

    server_attempts = 0
    timeout_attempts = 0

    while True:
        try:
            return fn()
        except Exception as exc:
            category, retry_after = classify(exc)

            if category == "client":
                raise

            if category == "timeout":
                timeout_attempts += 1
                if timeout_attempts >= config.max_attempts_timeout:
                    raise
                attempt = timeout_attempts
                limit = config.max_attempts_timeout
            else:
                server_attempts += 1
                if server_attempts >= config.max_attempts_server:
                    raise
                attempt = server_attempts
                limit = config.max_attempts_server

            if retry_after is not None and retry_after > 0:
                delay = retry_after
            else:
                delay = min(
                    config.base_delay_s * (2 ** (attempt - 1)),
                    config.max_delay_s,
                )
            delay += random.uniform(0, config.jitter_max_s)

            logger.warning(
                "Retry %d/%d (%s) after %.1fs: %s",
                attempt, limit, category, delay, exc,
            )
            time.sleep(delay)
