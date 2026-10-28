"""
detection/head_pose.py
──────────────────────
Head pose estimation via OpenCV solvePnP + MediaPipe landmarks.

Method:
  We use the classic 6-point PnP (Perspective-n-Point) approach:
    1. Define a canonical 3D face model in object space (cm).
    2. Select the corresponding 2D image-space landmark projections.
    3. Solve for the rotation vector (rvec) and translation vector (tvec).
    4. Decompose rvec → Euler angles (pitch, yaw, roll).

  This gives us head orientation in degrees relative to the camera axis,
  which is robust to face scale and position changes.

EMA Smoothing:
  Raw Euler angles jitter between frames due to landmark noise.
  We apply exponential moving average (EMA) per angle to smooth output
  without introducing significant lag.

Coordinate conventions:
  - Yaw   : + = right turn, - = left turn
  - Pitch : + = looking up, - = looking down
  - Roll  : + = right tilt, - = left tilt
"""

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple

from settings import HeadPoseConfig
from face_monitor import FaceState


# ── Canonical 3D face model points (in mm, object space) ───────────────────
# Derived from an anthropometric average adult face model.
# Order must match IMAGE_POINT_INDICES below.
FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0  ],   # Nose tip (origin)
    [0.0,   -63.6, -12.5 ],   # Chin
    [-43.3,  32.7, -26.0 ],   # Left eye outer corner
    [43.3,   32.7, -26.0 ],   # Right eye outer corner
    [-28.9, -28.9, -24.1 ],   # Left mouth corner
    [28.9,  -28.9, -24.1 ],   # Right mouth corner
], dtype=np.float64)

# MediaPipe landmark indices for the 6 selected points
# (must match FACE_3D_MODEL row order)
IMAGE_POINT_INDICES = [1, 152, 33, 263, 61, 291]


@dataclass
class HeadPoseResult:
    """Output of head pose estimation for a single frame."""
    valid: bool = False

    # Raw estimated angles (degrees)
    yaw_raw: float = 0.0
    pitch_raw: float = 0.0
    roll_raw: float = 0.0

    # EMA-smoothed angles — use these for decisions
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    # Derived behavioral flags (set by pose analyzer)
    looking_left: bool = False
    looking_right: bool = False
    looking_down: bool = False
    looking_up: bool = False
    tilted: bool = False

    # For overlay drawing: projected 3D axis endpoints
    nose_tip_2d: Optional[Tuple[int, int]] = None
    axis_points: Optional[np.ndarray] = None    # shape (3,2) for X/Y/Z axes


class HeadPoseEstimator:
    """
    Estimates head orientation from MediaPipe face landmarks using solvePnP.

    Designed for real-time use: inference is fast (< 1ms) since solvePnP
    is a closed-form algebraic solver, not a neural network.
    """

    def __init__(self, cfg: HeadPoseConfig, frame_h: int, frame_w: int):
        self.cfg = cfg
        self.frame_h = frame_h
        self.frame_w = frame_w

        # ── Camera intrinsics (approximate, assuming no distortion) ──────
        # Focal length heuristic: f ≈ frame_width for typical webcams
        focal = float(frame_w)
        cx = frame_w / 2.0
        cy = frame_h / 2.0
        self._camera_matrix = np.array([
            [focal, 0,    cx],
            [0,    focal, cy],
            [0,    0,     1 ],
        ], dtype=np.float64)
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # no lens distortion

        # ── EMA state (per-angle running average) ────────────────────────
        self._ema_yaw: float = 0.0
        self._ema_pitch: float = 0.0
        self._ema_roll: float = 0.0
        self._initialized: bool = False

        # For 3D axis projection
        self._axis_3d = np.float32([
            [50, 0, 0], [0, 50, 0], [0, 0, 50]   # X(red), Y(green), Z(blue)
        ])

    # ── Main entry point ────────────────────────────────────────────────

    def estimate(self, face_state: FaceState) -> HeadPoseResult:
        """
        Compute head pose from the current FaceState.

        Args:
            face_state: Output of FaceMonitor.process()

        Returns:
            HeadPoseResult with smoothed angles and behavioral flags.
        """
        result = HeadPoseResult()

        if not face_state.detected or face_state.landmarks is None:
            # Carry EMA state forward but mark as invalid
            result.yaw = self._ema_yaw
            result.pitch = self._ema_pitch
            result.roll = self._ema_roll
            return result

        lm = face_state.landmarks  # shape (N, 3), normalized [0,1]

        # ── Extract 2D image points for selected landmarks ────────────
        image_points = np.array([
            [lm[i][0] * self.frame_w, lm[i][1] * self.frame_h]
            for i in IMAGE_POINT_INDICES
        ], dtype=np.float64)

        # ── solvePnP: finds rvec/tvec that maps 3D model → 2D image ──
        success, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return result

        # ── Rotation matrix from Rodrigues vector ─────────────────────
        rmat, _ = cv2.Rodrigues(rvec)

        # ── Decompose to Euler angles ─────────────────────────────────
        # RQDecomp3x3 returns angles in degrees directly
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch_raw = angles[0]
        yaw_raw   = angles[1]
        roll_raw  = angles[2]

        # Fix Euler angle wrap-around for pitch (it hovers near -180 / +180)
        # If it flips to a positive number, subtract 360 to keep it continuously negative
        if pitch_raw > 0:
            pitch_raw -= 360

        # ── Apply EMA smoothing ───────────────────────────────────────
        alpha = self.cfg.ema_alpha
        if not self._initialized:
            self._ema_yaw   = yaw_raw
            self._ema_pitch = pitch_raw
            self._ema_roll  = roll_raw
            self._initialized = True
        else:
            self._ema_yaw   = alpha * yaw_raw   + (1 - alpha) * self._ema_yaw
            self._ema_pitch = alpha * pitch_raw + (1 - alpha) * self._ema_pitch
            self._ema_roll  = alpha * roll_raw  + (1 - alpha) * self._ema_roll

        # ── Populate result ───────────────────────────────────────────
        result.valid = True
        result.yaw_raw, result.pitch_raw, result.roll_raw = yaw_raw, pitch_raw, roll_raw
        result.yaw   = self._ema_yaw
        result.pitch = self._ema_pitch
        result.roll  = self._ema_roll

        # ── Behavioral classification ─────────────────────────────────
        result.looking_left  = result.yaw < -self.cfg.yaw_alert_threshold
        result.looking_right = result.yaw >  self.cfg.yaw_alert_threshold
        result.looking_down  = result.pitch > self.cfg.pitch_down_threshold
        result.looking_up    = result.pitch < self.cfg.pitch_up_threshold
        result.tilted        = abs(result.roll) > self.cfg.roll_alert_threshold

        # ── Project 3D axes for overlay drawing ───────────────────────
        try:
            nose_2d = image_points[0]    # nose tip in image space
            axis_pts, _ = cv2.projectPoints(
                self._axis_3d, rvec, tvec,
                self._camera_matrix, self._dist_coeffs,
            )
            result.nose_tip_2d = (int(nose_2d[0]), int(nose_2d[1]))
            result.axis_points = axis_pts.reshape(-1, 2).astype(int)
        except Exception:
            pass  # non-critical — overlay skipped

        return result

    # ── Drawing helper ──────────────────────────────────────────────────

    def draw_pose_axes(
        self,
        frame: np.ndarray,
        pose: HeadPoseResult,
        axis_length: int = 60,
    ) -> None:
        """
        Draw RGB XYZ axes projected from nose tip to visualize head orientation.
        Red=X, Green=Y, Blue=Z
        """
        if not pose.valid or pose.nose_tip_2d is None or pose.axis_points is None:
            return

        origin = pose.nose_tip_2d
        ax_pts = pose.axis_points

        cv2.arrowedLine(frame, origin, tuple(ax_pts[0]), (0, 0, 200),   2, tipLength=0.3)  # X red
        cv2.arrowedLine(frame, origin, tuple(ax_pts[1]), (0, 200, 0),   2, tipLength=0.3)  # Y green
        cv2.arrowedLine(frame, origin, tuple(ax_pts[2]), (200, 100, 0), 2, tipLength=0.3)  # Z blue

    def draw_pose_text(
        self,
        frame: np.ndarray,
        pose: HeadPoseResult,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """Overlay yaw/pitch/roll values as text."""
        if not pose.valid:
            return
        x, y = position
        cv2.putText(frame,
                    f"Yaw:{pose.yaw:+.1f} Pitch:{pose.pitch:+.1f} Roll:{pose.roll:+.1f}",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
