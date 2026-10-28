"""
events/handler.py
─────────────────
Evidence capture and event management.

Responsibilities:
  1. Maintain a circular frame buffer (last N seconds of video).
  2. On cheating alert: save screenshot + video clip from buffer.
  3. Generate structured event records with timestamps.
  4. Prevent disk flooding via event deduplication.

Circular Buffer Design:
  We use a collections.deque with maxlen = fps * buffer_seconds.
  Each element stores (timestamp, bgr_frame). On alert, we dump the
  deque to a video file — this gives us the "last N seconds before"
  the alert fired, which is the most forensically useful window.
"""

from __future__ import annotations
import cv2
import time
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Deque, Tuple
import numpy as np

from settings import EventConfig


@dataclass
class CheatingEvent:
    """Immutable record of a detected cheating incident."""
    event_id: str
    timestamp: float
    reason: str                          # human-readable trigger description
    suspicion_score: float
    screenshot_path: Optional[str] = None
    video_clip_path: Optional[str] = None
    active_signals: dict = field(default_factory=dict)


class EventHandler:
    """
    Manages the evidence capture pipeline for detected cheating events.

    Thread safety: NOT thread-safe. Call from main video loop only.
    File I/O (video writing) is synchronous — for production systems,
    offload to a background thread/process.
    """

    def __init__(self, cfg: EventConfig, fps: float = 15.0):
        self.cfg = cfg
        self.fps = fps

        # ── Circular frame buffer ─────────────────────────────────────
        max_frames = int(fps * cfg.video_buffer_seconds)
        self._frame_buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=max_frames)

        # ── Event state ───────────────────────────────────────────────
        self._events: List[CheatingEvent] = []
        self._event_counter: int = 0
        self._last_save_time: float = 0.0

        # ── Output dir setup ──────────────────────────────────────────
        os.makedirs(cfg.capture_dir, exist_ok=True)

    # ── Public API ───────────────────────────────────────────────────────

    def push_frame(self, frame: np.ndarray):
        """
        Add a frame to the circular buffer.
        Call every processed frame regardless of alert state.
        """
        self._frame_buffer.append((time.time(), frame.copy()))

    def handle_alert(
        self,
        reason: str,
        suspicion_score: float,
        active_signals: dict,
        current_frame: np.ndarray,
    ) -> Optional[CheatingEvent]:
        """
        Handle a confirmed cheating detection event.

        Saves screenshot and video clip. Returns a CheatingEvent record.
        Skips if called too soon after last save (deduplication).
        """
        now = time.time()

        # Deduplication: don't create multiple events within a short window
        # (separate from alert cooldown in scorer — this is filesystem-level)
        if now - self._last_save_time < 5.0:
            return None

        self._last_save_time = now
        self._event_counter += 1
        event_id = f"EVT_{int(now)}_{self._event_counter:04d}"
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))

        screenshot_path = None
        video_path = None

        # ── Screenshot ────────────────────────────────────────────────
        if self.cfg.screenshot_on_alert:
            screenshot_path = os.path.join(
                self.cfg.capture_dir, f"{event_id}_screenshot_{timestamp_str}.jpg"
            )
            annotated = self._annotate_screenshot(current_frame.copy(), reason, suspicion_score)
            cv2.imwrite(screenshot_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # ── Video clip from circular buffer ───────────────────────────
        if self.cfg.video_clip_on_alert and len(self._frame_buffer) > 0:
            video_path = os.path.join(
                self.cfg.capture_dir, f"{event_id}_clip_{timestamp_str}.mp4"
            )
            self._save_video_clip(video_path)

        event = CheatingEvent(
            event_id=event_id,
            timestamp=now,
            reason=reason,
            suspicion_score=suspicion_score,
            screenshot_path=screenshot_path,
            video_clip_path=video_path,
            active_signals=dict(active_signals),
        )
        self._events.append(event)
        return event

    # ── Internal helpers ──────────────────────────────────────────────────

    def _annotate_screenshot(
        self,
        frame: np.ndarray,
        reason: str,
        score: float,
    ) -> np.ndarray:
        """Add alert banner and metadata to screenshot."""
        h, w = frame.shape[:2]

        # Red alert banner at top
        cv2.rectangle(frame, (0, 0), (w, 45), (0, 0, 180), -1)
        cv2.putText(
            frame,
            f"MALPRACTICE ALERT  Score:{score:.1f}  {reason}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Timestamp at bottom
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(
            frame, f"Captured: {ts}", (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )
        return frame

    def _save_video_clip(self, path: str):
        """Write buffered frames to an MP4 file."""
        if not self._frame_buffer:
            return

        frames = list(self._frame_buffer)
        if not frames:
            return

        h, w = frames[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.cfg.clip_codec)
        writer = cv2.VideoWriter(path, fourcc, self.cfg.clip_fps, (w, h))

        for _, frame in frames:
            writer.write(frame)

        writer.release()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def recent_events(self) -> List[CheatingEvent]:
        return list(self._events[-10:])   # last 10 events

    @property
    def buffer_fill_ratio(self) -> float:
        return len(self._frame_buffer) / max(self._frame_buffer.maxlen, 1)
