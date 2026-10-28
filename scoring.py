"""
utils/scoring.py
────────────────
Suspicion scoring engine.

Design:
  Scores are accumulated from weighted behavioral signals and decay
  exponentially every frame. This creates a "forgetting" mechanism:
  brief misbehaviors raise the score but it quickly returns to baseline
  unless sustained. Only persistent or severe behavior crosses the
  alert threshold.

  The score is NOT a probability — it's an unnormalized risk index
  calibrated to the configured threshold.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from settings import SuspicionConfig


@dataclass
class ScoringEvent:
    """Record of a single scoring contribution."""
    timestamp: float
    signal_name: str
    weight: float
    score_before: float
    score_after: float


class SuspicionScorer:
    """
    Weighted, decay-based suspicion scoring system.

    Architecture:
      - Each behavioral signal contributes a weight to the score.
      - Every frame, the score is multiplied by decay_rate (<1), so it
        naturally returns to zero when signals stop firing.
      - An alert is raised when score >= alert_threshold.
      - Cooldown prevents repeated alerts for the same sustained event.

    Thread safety: NOT thread-safe. Call only from the main video loop.
    """

    def __init__(self, cfg: SuspicionConfig):
        self.cfg = cfg
        self.score: float = 0.0
        self._last_alert_time: float = 0.0
        self._alert_active: bool = False
        self._history: List[ScoringEvent] = []
        self._frame_contributions: Dict[str, float] = {}  # this-frame debug info

    # ── Public API ─────────────────────────────────────────────────────────

    def apply_decay(self):
        """
        Call once per frame BEFORE accumulating signal weights.
        Implements exponential decay: score → score * decay_rate
        """
        self.score *= self.cfg.score_decay_rate
        self.score = max(self.score, 0.0)
        self._frame_contributions.clear()

    def add_signal(self, signal_name: str, weight: float, active: bool):
        """
        Conditionally add a weighted signal to the current frame's score.

        Args:
            signal_name : Identifier for logging/debug.
            weight      : Score contribution when active.
            active      : Whether the signal is firing this frame.
        """
        if not active:
            return

        before = self.score
        contribution = weight * active  # weight if active, 0 if not
        self.score = min(self.score + contribution, self.cfg.max_score)
        self._frame_contributions[signal_name] = contribution

        self._history.append(ScoringEvent(
            timestamp=time.time(),
            signal_name=signal_name,
            weight=weight,
            score_before=before,
            score_after=self.score,
        ))

        # Trim history to last 500 events to avoid unbounded memory growth
        if len(self._history) > 500:
            self._history = self._history[-500:]

    def evaluate(self) -> tuple[bool, bool]:
        """
        Evaluate current score against thresholds.

        Returns:
            (is_alert, is_warning) — both booleans.
            is_alert is True only once per cooldown period.
        """
        is_warning = self.score >= self.cfg.warning_threshold
        is_alert = False

        if self.score >= self.cfg.alert_threshold:
            now = time.time()
            if now - self._last_alert_time >= self.cfg.alert_cooldown_sec:
                self._last_alert_time = now
                is_alert = True
                self._alert_active = True

        if self.score < self.cfg.warning_threshold:
            self._alert_active = False

        return is_alert, is_warning

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def normalized_score(self) -> float:
        """Score normalized to [0, 1] relative to alert threshold."""
        return min(self.score / self.cfg.alert_threshold, 1.0)

    @property
    def level(self) -> str:
        """Human-readable risk level string."""
        if self.score >= self.cfg.alert_threshold:
            return "ALERT"
        elif self.score >= self.cfg.warning_threshold:
            return "WARNING"
        else:
            return "OK"

    @property
    def active_signals(self) -> Dict[str, float]:
        """Signals that contributed this frame."""
        return dict(self._frame_contributions)

    def get_recent_events(self, last_n_seconds: float = 30.0) -> List[ScoringEvent]:
        """Return events within the last N seconds for audit trail."""
        cutoff = time.time() - last_n_seconds
        return [e for e in self._history if e.timestamp >= cutoff]

    def reset(self):
        self.score = 0.0
        self._alert_active = False
        self._frame_contributions.clear()

    def __repr__(self) -> str:
        return (f"SuspicionScorer(score={self.score:.2f}, "
                f"level={self.level}, active={list(self._frame_contributions.keys())})")
