"""
utils/overlay.py
────────────────
Professional HUD (Heads-Up Display) overlay renderer.

Renders all monitoring information onto the video frame:
  - Suspicion score bar with color gradient
  - Frame counter bars for each behavior signal
  - Head pose angles
  - FPS and pipeline timing
  - Alert/Warning banners
  - Event count and buffer status
"""

from __future__ import annotations
import cv2
import numpy as np
import time
from typing import Dict, Optional, Tuple

from settings import DisplayConfig, SuspicionConfig
from face_monitor import FaceState
from head_pose import HeadPoseResult
from phone_detector import PhoneDetectionResult
from scoring import SuspicionScorer
from temporal import TemporalStateManager
from performance import FPSTracker


class HUDRenderer:
    """
    All overlay drawing logic centralized here to keep the main loop clean.

    Uses OpenCV drawing primitives with alpha blending for semi-transparent
    panels, colored bars, and alert banners.
    """

    def __init__(self, cfg: DisplayConfig, sus_cfg: SuspicionConfig):
        self.cfg = cfg
        self.sus_cfg = sus_cfg
        self._alert_flash_state: bool = False
        self._alert_flash_timer: float = 0.0

    # ── Main render entry point ──────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        face_state: FaceState,
        pose: HeadPoseResult,
        phone_result: PhoneDetectionResult,
        scorer: SuspicionScorer,
        tsm: TemporalStateManager,
        fps_tracker: FPSTracker,
        event_count: int,
        stage_report: str = "",
    ) -> np.ndarray:
        """
        Draw all HUD elements onto the frame in-place.
        Returns the frame with overlays applied.
        """
        h, w = frame.shape[:2]

        # ── Semi-transparent left panel background ────────────────────
        self._draw_panel(frame, x=0, y=0, width=280, height=h, alpha=0.45)

        y_cursor = 25  # running y-position for left panel items

        # ── FPS ───────────────────────────────────────────────────────
        if self.cfg.show_fps:
            fps_color = self.cfg.color_ok if fps_tracker.fps > 20 else self.cfg.color_warning
            cv2.putText(frame, f"FPS: {fps_tracker.fps:.1f}",
                        (10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX,
                        self.cfg.font_scale_large, fps_color, self.cfg.font_thickness, cv2.LINE_AA)
            y_cursor += 28

        # ── Suspicion score bar ───────────────────────────────────────
        if self.cfg.show_score_bar:
            y_cursor = self._draw_score_bar(frame, scorer, y_cursor)
            y_cursor += 8

        # ── Frame counter bars ────────────────────────────────────────
        if self.cfg.show_frame_counters:
            y_cursor = self._draw_counter_bars(frame, tsm, y_cursor)
            y_cursor += 8

        # ── Head pose readout ─────────────────────────────────────────
        y_cursor = self._draw_pose_readout(frame, pose, y_cursor)
        y_cursor += 8

        # ── Face status ───────────────────────────────────────────────
        y_cursor = self._draw_face_status(frame, face_state, y_cursor)

        # ── Phone status ──────────────────────────────────────────────
        y_cursor = self._draw_phone_status(frame, phone_result, y_cursor)

        # ── Event counter (bottom-left) ───────────────────────────────
        cv2.putText(frame, f"Events: {event_count}",
                    (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX,
                    self.cfg.font_scale_small, self.cfg.color_info, 1, cv2.LINE_AA)

        # ── Pipeline timing (bottom-left) ─────────────────────────────
        if stage_report:
            cv2.putText(frame, stage_report,
                        (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, self.cfg.color_info, 1, cv2.LINE_AA)

        # ── Alert / Warning banner ────────────────────────────────────
        self._draw_alert_banner(frame, scorer, w)

        # ── Behavior flag pills (top-right) ───────────────────────────
        self._draw_behavior_pills(frame, pose, face_state, phone_result, w)

        return frame

    # ── Component renderers ──────────────────────────────────────────────

    def _draw_score_bar(self, frame: np.ndarray, scorer: SuspicionScorer, y: int) -> int:
        """Render the suspicion score as a colored fill bar."""
        label = f"SUSPICION  {scorer.score:.1f}/{self.sus_cfg.alert_threshold:.0f}"
        cv2.putText(frame, label, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, self.cfg.font_scale_small,
                    self.cfg.color_info, 1, cv2.LINE_AA)
        y += 16

        bx = self.cfg.score_bar_x
        bw = self.cfg.score_bar_width
        bh = self.cfg.score_bar_height

        # Background bar
        cv2.rectangle(frame, (bx, y), (bx + bw, y + bh), (60, 60, 60), -1)

        # Fill bar — color interpolated from green → orange → red
        fill_w = int(bw * scorer.normalized_score)
        if fill_w > 0:
            bar_color = self._score_color(scorer.normalized_score)
            cv2.rectangle(frame, (bx, y), (bx + fill_w, y + bh), bar_color, -1)

        # Threshold marker line
        thresh_x = bx + int(bw * (self.sus_cfg.warning_threshold / self.sus_cfg.alert_threshold))
        cv2.line(frame, (thresh_x, y - 2), (thresh_x, y + bh + 2), self.cfg.color_warning, 1)

        # Border
        cv2.rectangle(frame, (bx, y), (bx + bw, y + bh), (120, 120, 120), 1)

        # Level label
        level_color = {
            "OK": self.cfg.color_ok,
            "WARNING": self.cfg.color_warning,
            "ALERT": self.cfg.color_alert,
        }.get(scorer.level, self.cfg.color_info)

        cv2.putText(frame, scorer.level,
                    (bx + bw + 8, y + bh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, level_color, 2, cv2.LINE_AA)

        return y + bh + 4

    def _draw_counter_bars(self, frame: np.ndarray, tsm: TemporalStateManager, y: int) -> int:
        """Draw mini fill bars for each temporal counter."""
        state = tsm.get_state()
        bar_w = 180
        bar_h = 10
        x = 10

        COUNTER_LABELS = {
            "look_away":   "Look Away",
            "look_down":   "Look Down",
            "face_absent": "No Face",
            "phone":       "Phone",
        }

        for name, data in state.items():
            label = COUNTER_LABELS.get(name, name)
            color = self.cfg.color_alert if data["triggered"] else self.cfg.color_info

            cv2.putText(frame, f"{label} ({data['value']:.0f})",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            y += 13

            # Background
            cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)

            # Fill
            fill = int(bar_w * data["fill"])
            if fill > 0:
                fill_color = self.cfg.color_alert if data["triggered"] else (80, 130, 200)
                cv2.rectangle(frame, (x, y), (x + fill, y + bar_h), fill_color, -1)

            cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (100, 100, 100), 1)
            y += bar_h + 6

        return y

    def _draw_pose_readout(self, frame: np.ndarray, pose: HeadPoseResult, y: int) -> int:
        """Render Yaw/Pitch/Roll with directional color coding."""
        cv2.putText(frame, "HEAD POSE",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.cfg.color_accent, 1, cv2.LINE_AA)
        y += 16

        if not pose.valid:
            cv2.putText(frame, "  No valid pose", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.cfg.color_info, 1, cv2.LINE_AA)
            return y + 16

        entries = [
            ("Yaw",   pose.yaw,   pose.looking_left or pose.looking_right),
            ("Pitch", pose.pitch, pose.looking_down or pose.looking_up),
            ("Roll",  pose.roll,  pose.tilted),
        ]
        for label, val, flagged in entries:
            color = self.cfg.color_warning if flagged else self.cfg.color_info
            cv2.putText(frame, f"  {label}: {val:+.1f}°",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y += 16

        return y

    def _draw_face_status(self, frame: np.ndarray, face_state: FaceState, y: int) -> int:
        cv2.putText(frame, "FACE STATUS",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.cfg.color_accent, 1, cv2.LINE_AA)
        y += 16

        if face_state.detected:
            color = self.cfg.color_ok
            txt = f"  Present ({face_state.num_faces} face{'s' if face_state.num_faces > 1 else ''})"
            if face_state.multiple_faces:
                color = self.cfg.color_alert
        else:
            color = self.cfg.color_alert
            absent = face_state.face_absent_seconds
            txt = f"  ABSENT {absent:.1f}s"

        cv2.putText(frame, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += 16

        if face_state.centroid_velocity > 0:
            vel_color = self.cfg.color_warning if face_state.centroid_velocity > 30 else self.cfg.color_info
            cv2.putText(frame, f"  Velocity: {face_state.centroid_velocity:.1f}px",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, vel_color, 1, cv2.LINE_AA)
            y += 14

        return y

    def _draw_phone_status(self, frame: np.ndarray, phone_result: PhoneDetectionResult, y: int) -> int:
        cv2.putText(frame, "PHONE",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.cfg.color_accent, 1, cv2.LINE_AA)
        y += 16

        if phone_result.phone_detected:
            best = phone_result.best
            color = self.cfg.color_alert
            cv2.putText(frame, f"  DETECTED conf={best.confidence:.2f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "  Not detected",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.cfg.color_info, 1, cv2.LINE_AA)

        return y + 16

    def _draw_alert_banner(self, frame: np.ndarray, scorer: SuspicionScorer, w: int):
        """Full-width banner at top for ALERT/WARNING states."""
        if scorer.level == "ALERT":
            # Flashing red banner
            now = time.time()
            if now - self._alert_flash_timer > 0.4:
                self._alert_flash_state = not self._alert_flash_state
                self._alert_flash_timer = now

            if self._alert_flash_state:
                cv2.rectangle(frame, (280, 0), (w, 48), (0, 0, 180), -1)
                cv2.putText(frame, "⚠  MALPRACTICE ALERT  ⚠",
                            (320, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 255), 2, cv2.LINE_AA)

        elif scorer.level == "WARNING":
            cv2.rectangle(frame, (280, 0), (w, 36), (0, 90, 180), -1)
            cv2.putText(frame, "SUSPICIOUS BEHAVIOR — MONITORING",
                        (290, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (255, 230, 180), 1, cv2.LINE_AA)

    def _draw_behavior_pills(
        self,
        frame: np.ndarray,
        pose: HeadPoseResult,
        face_state: FaceState,
        phone_result: PhoneDetectionResult,
        frame_w: int,
    ):
        """Small colored status pills in top-right corner."""
        pills = []
        if pose.looking_left:  pills.append(("◄ LEFT",  self.cfg.color_warning))
        if pose.looking_right: pills.append(("RIGHT ►", self.cfg.color_warning))
        if pose.looking_down:  pills.append(("▼ DOWN",  self.cfg.color_warning))
        if not face_state.detected: pills.append(("NO FACE", self.cfg.color_alert))
        if face_state.multiple_faces: pills.append(("MULTI FACE", self.cfg.color_alert))
        if phone_result.phone_detected: pills.append(("PHONE!", self.cfg.color_alert))

        x_right = frame_w - 10
        y = 55
        for text, color in pills:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            x = x_right - tw - 10
            cv2.rectangle(frame, (x - 5, y - th - 4), (x_right, y + 4), color, -1)
            cv2.putText(frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            y += th + 12

    # ── Utility helpers ──────────────────────────────────────────────────

    @staticmethod
    def _draw_panel(frame: np.ndarray, x: int, y: int, width: int, height: int, alpha: float):
        """Draw semi-transparent dark panel via alpha blending."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def _score_color(normalized: float) -> Tuple[int, int, int]:
        """Interpolate BGR color from green (0) → orange (0.5) → red (1.0)."""
        if normalized < 0.5:
            t = normalized * 2
            b = 0
            g = int(200 * (1 - t) + 165 * t)
            r = int(50 * (1 - t) + 0 * t)
            return (b, g, r)
        else:
            t = (normalized - 0.5) * 2
            b = 0
            g = int(165 * (1 - t))
            r = int(0 * (1 - t) + 220 * t)
            return (b, g, r)
