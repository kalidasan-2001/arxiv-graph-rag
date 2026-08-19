"""Structured (text-formatted) logging configuration.

Standard library ``logging`` is sufficient for this stage. Each record
includes a timestamp, logger name, level and message. More advanced
observability (OpenTelemetry, log shipping, etc.) is deferred.
"""

import logging
import sys

from app.core.config import Settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(settings: Settings) -> None:
    """Configure the root logger for the application.

    Idempotent: safe to call multiple times (e.g. across tests) without
    duplicating log handlers.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL.upper())

    # Avoid attaching duplicate handlers if called more than once.
    root_logger.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root_logger.addHandler(handler)
