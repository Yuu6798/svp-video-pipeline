"""Structured logging helpers for CLI verbose mode and log.json writes."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any


def setup_verbose_logger() -> logging.Logger:
    """Set up stdout JSON logging for --verbose mode."""
    logger = logging.getLogger("svp_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        extra_data = getattr(record, "extra_data", None)
        if isinstance(extra_data, dict):
            payload.update(extra_data)
        return json.dumps(payload, ensure_ascii=False)


def write_log_json(log_path: Path, data: dict[str, Any]) -> None:
    """Write a structured JSON log file."""
    log_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
