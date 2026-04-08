"""
Telemetry & Structured Logging

Provides:
  - Structured log lines (JSON-compatible for log aggregators)
  - Per-session metrics: token usage, total latency, step timing
  - System-latency display string for the Gradio UI footer

Keep this module import-safe — no heavy deps, no circular imports.
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Logger setup ──────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ── Session metrics ───────────────────────────────────────────────────────────

@dataclass
class SessionMetrics:
    session_id:     str   = field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d-%H%M%S"))
    total_tokens:   int   = 0
    total_latency_ms: float = 0.0
    api_calls:      int   = 0
    api_errors:     int   = 0
    facts_extracted: int  = 0
    human_corrections: int = 0
    _start_time:    float = field(default_factory=time.perf_counter)

    def record_call(self, tokens: int, latency_ms: float, error: bool = False) -> None:
        self.total_tokens    += tokens
        self.total_latency_ms += latency_ms
        self.api_calls       += 1
        if error:
            self.api_errors += 1

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.api_calls, 1)

    @property
    def session_duration_s(self) -> float:
        return round(time.perf_counter() - self._start_time, 1)

    def status_line(self) -> str:
        return (
            f"session {self.session_id} │ "
            f"calls={self.api_calls} │ "
            f"tokens={self.total_tokens:,} │ "
            f"avg_latency={self.avg_latency_ms:.0f}ms │ "
            f"errors={self.api_errors} │ "
            f"facts={self.facts_extracted} │ "
            f"uptime={self.session_duration_s}s"
        )

    def ui_footer(self) -> str:
        """Short string for the Gradio UI status bar."""
        return (
            f"⚡ {self.avg_latency_ms:.0f}ms avg latency  "
            f"│  🔤 {self.total_tokens:,} tokens used  "
            f"│  📡 {self.api_calls} API calls  "
            f"│  🧠 {self.facts_extracted} facts learned"
        )


# Global session metrics (reset per run)
_session: Optional[SessionMetrics] = None

def get_session() -> SessionMetrics:
    global _session
    if _session is None:
        _session = SessionMetrics()
    return _session

def reset_session() -> SessionMetrics:
    global _session
    _session = SessionMetrics()
    return _session
