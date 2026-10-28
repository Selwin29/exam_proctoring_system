"""
utils/logger.py
───────────────
Structured event logger with rate-limiting.

Design goals:
  - Rate limiting per event type: prevents log spam when a condition
    is true for hundreds of consecutive frames.
  - Severity levels: DEBUG / INFO / WARNING / ALERT / CRITICAL
  - Writes to both file and stderr (optionally).
  - Each event is a structured dict → easy to parse for analytics.
"""

from __future__ import annotations
import logging
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any


class RateLimitedLogger:
    """
    Wraps Python's logging.Logger with per-event-type rate limiting.

    Usage:
        logger = RateLimitedLogger("proctoring", "logs/proctoring.log")
        logger.event("look_away", "Candidate looking left", severity="WARNING",
                     data={"yaw": 32.5, "score": 7.2})
    """

    SEVERITY_MAP = {
        "DEBUG":    logging.DEBUG,
        "INFO":     logging.INFO,
        "WARNING":  logging.WARNING,
        "ALERT":    logging.WARNING,     # map ALERT → WARNING in Python logging
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(
        self,
        name: str,
        log_file: str,
        min_interval_sec: float = 2.0,
        console_output: bool = True,
    ):
        self.min_interval_sec = min_interval_sec
        self._last_log_time: Dict[str, float] = defaultdict(float)

        # ── Configure Python logger ───────────────────────────────────
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler — JSON-structured lines
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

        # Console handler
        if console_output:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)

    # ── Public API ──────────────────────────────────────────────────────

    def event(
        self,
        event_type: str,
        message: str,
        severity: str = "INFO",
        data: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> bool:
        """
        Log an event, respecting the rate limit for this event_type.

        Args:
            event_type : Unique key for rate-limiting (e.g. "look_away").
            message    : Human-readable description.
            severity   : One of DEBUG/INFO/WARNING/ALERT/CRITICAL.
            data       : Optional structured payload for analytics.
            force      : Bypass rate limiting (for critical one-time events).

        Returns:
            True if the event was actually logged.
        """
        now = time.time()
        if not force and (now - self._last_log_time[event_type]) < self.min_interval_sec:
            return False

        self._last_log_time[event_type] = now
        level = self.SEVERITY_MAP.get(severity.upper(), logging.INFO)

        payload = {
            "event": event_type,
            "message": message,
            "ts": now,
        }
        if data:
            payload["data"] = data

        self._logger.log(level, json.dumps(payload))
        
        # ── FLASK INTEGRATION: EMIT EVENT TO STDOUT ──
        flask_sev = "info"
        s_upper = severity.upper()
        if s_upper in ["ALERT", "CRITICAL"]:
            flask_sev = "high"
        elif s_upper == "WARNING":
            flask_sev = "medium"
            
        flask_evt = {
            "type": event_type,
            "detail": message,
            "severity": flask_sev
        }
        print(json.dumps(flask_evt), flush=True)
        
        return True

    def alert(self, event_type: str, message: str, data: Optional[Dict] = None, force: bool = True):
        self.event(event_type, message, severity="ALERT", data=data, force=force)

    def info(self, event_type: str, message: str, data: Optional[Dict] = None, force: bool = False):
        self.event(event_type, message, severity="INFO", data=data, force=force)

    def warning(self, event_type: str, message: str, data: Optional[Dict] = None, force: bool = False):
        self.event(event_type, message, severity="WARNING", data=data, force=force)

    def debug(self, event_type: str, message: str, data: Optional[Dict] = None, force: bool = False):
        self.event(event_type, message, severity="DEBUG", data=data, force=force)
        
    def error(self, event_type: str, message: str, data: Optional[Dict] = None, force: bool = False):
        self.event(event_type, message, severity="CRITICAL", data=data, force=force)

    def reset_rate_limit(self, event_type: str):
        """Force the next log of this type to pass through regardless of cooldown."""
        self._last_log_time[event_type] = 0.0
