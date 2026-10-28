"""
detection/face_monitor.py
─────────────────────────
Face presence detection and tracking via MediaPipe Face Mesh.

Responsibilities:
  - Run Face Mesh inference on each frame
  - Track face centroid stability (rapid head movement detection)
  - Detect face disappearance with timeout logic
  - Detect multiple faces (potential accomplice present)
  - Expose raw landmark data for downstream head-pose estimation

MediaPipe Face Mesh produces 468 3D landmarks per face.
We use a subset for pose estimation, iris tracking, and presence checks.
"""

from __future__ import annotations
import time
import numpy as np
import cv2
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from settings import FaceMeshConfig


# ── Landmark indices used downstream ───────────────────────────────────────
# These are stable across MediaPipe versions for the canonical 468-landmark model
NOSE_TIP       = 1
CHIN           = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH     = 61
RIGHT_MOUTH    = 291
LEFT_IRIS_CENTER  = 468   # only available with refine_landmarks=True
RIGHT_IRIS_CENTER = 473


@dataclass
class FaceState:
    """
    Snapshot of face detection results for a single frame.
    Passed between pipeline stages to avoid redundant computation.
    """
    detected: bool = False
    num_faces: int = 0

    # Normalized face bounding box [0,1] — derived from landmark extents
    bbox_norm: Optional[Tuple[float, float, float, float]] = None  # x1,y1,x2,y2

    # All normalized bounding boxes (for multi-face tracking)
    all_bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)

    # Face centroid in pixel coordinates
    centroid_px: Optional[Tuple[int, int]] = None

    # Raw landmark array (N x 3, normalized) for the primary face
    landmarks: Optional[np.ndarray] = None

    # 3D model points for solvePnP (populated by head pose module)
    model_points: Optional[np.ndarray] = None

    # Velocity of centroid from previous frame (pixels/frame)
    centroid_velocity: float = 0.0

    # Timestamp of last successful detection
    last_seen: float = field(default_factory=time.time)

    # Alert flags
    face_absent_seconds: float = 0.0
    multiple_faces: bool = False


class FaceMonitor:
    """
    Wraps MediaPipe FaceMesh with state tracking and alerting logic.

    Design decisions:
      - We instantiate FaceMesh once and reuse across frames (expensive init).
      - Landmark coordinates are normalized [0,1] by MediaPipe; we convert
        to pixels only when needed for OpenCV drawing.
      - The centroid is computed as the mean of all 468 landmarks (more robust
        than a specific point against partial occlusion).
    """

    def __init__(self, cfg: FaceMeshConfig, frame_h: int, frame_w: int):
        self.cfg = cfg
        self.frame_h = frame_h
        self.frame_w = frame_w

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=cfg.max_num_faces,
            refine_landmarks=cfg.refine_landmarks,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

        self._prev_centroid: Optional[Tuple[float, float]] = None
        self._last_detection_time: float = time.time()
        self._state = FaceState()

    # ── Main entry point ────────────────────────────────────────────────

    def process(self, rgb_frame: np.ndarray) -> FaceState:
        """
        Run face mesh inference and update internal state.

        Args:
            rgb_frame : Frame in RGB (MediaPipe requires RGB, not BGR).

        Returns:
            Updated FaceState snapshot.
        """
        results = self._face_mesh.process(rgb_frame)
        now = time.time()

        state = FaceState()
        state.last_seen = self._state.last_seen  # carry forward until confirmed

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            state.detected = True
            state.num_faces = len(results.multi_face_landmarks)
            state.multiple_faces = state.num_faces > 1
            state.last_seen = now
            self._last_detection_time = now

            # Primary face = first detection (assumed to be the exam candidate)
            primary = results.multi_face_landmarks[0]

            # ── Convert landmark list to numpy array ──────────────────
            lm_array = np.array(
                [(lm.x, lm.y, lm.z) for lm in primary.landmark],
                dtype=np.float32,
            )
            state.landmarks = lm_array

            # ── Bounding boxes from landmark extents ────────────────────
            for face in results.multi_face_landmarks:
                xs_all = [lm.x for lm in face.landmark]
                ys_all = [lm.y for lm in face.landmark]
                state.all_bboxes.append((
                    float(min(xs_all)), float(min(ys_all)),
                    float(max(xs_all)), float(max(ys_all))
                ))

            state.bbox_norm = state.all_bboxes[0]

            # ── Centroid in pixel space ───────────────────────────────
            xs = lm_array[:, 0]
            ys = lm_array[:, 1]
            cx = float(np.mean(xs)) * self.frame_w
            cy = float(np.mean(ys)) * self.frame_h
            state.centroid_px = (int(cx), int(cy))

            # ── Centroid velocity for rapid-movement detection ────────
            if self._prev_centroid is not None:
                dx = cx - self._prev_centroid[0]
                dy = cy - self._prev_centroid[1]
                state.centroid_velocity = float(np.sqrt(dx**2 + dy**2))
            self._prev_centroid = (cx, cy)

        else:
            # Face not detected this frame
            state.detected = False
            state.face_absent_seconds = now - self._last_detection_time
            self._prev_centroid = None  # reset velocity on loss

        self._state = state
        return state

    # ── Drawing helpers ─────────────────────────────────────────────────

    def draw_face_bbox(
        self,
        frame: np.ndarray,
        state: FaceState,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw face bounding box derived from landmark extents."""
        if not state.detected or state.bbox_norm is None:
            return

        x1n, y1n, x2n, y2n = state.bbox_norm
        x1 = int(x1n * self.frame_w)
        y1 = int(y1n * self.frame_h)
        x2 = int(x2n * self.frame_w)
        y2 = int(y2n * self.frame_h)

        # Rounded rectangle corners via multiple arcs (no native rounded rect in OpenCV)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Centroid marker
        if state.centroid_px:
            cv2.circle(frame, state.centroid_px, 4, color, -1)

    def draw_landmarks(
        self,
        frame: np.ndarray,
        state: FaceState,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """Draw all 468 face mesh landmarks as tiny dots."""
        if not state.detected or state.landmarks is None:
            return

        for lm in state.landmarks:
            px = int(lm[0] * self.frame_w)
            py = int(lm[1] * self.frame_h)
            cv2.circle(frame, (px, py), 1, color, -1)

    def release(self):
        self._face_mesh.close()
