"""
config/settings.py
──────────────────
Central configuration for the Agamotto.
All tunable hyper-parameters live here — no magic numbers in business logic.
"""

from dataclasses import dataclass, field
from typing import Tuple
import os


# ──────────────────────────────────────────────
# Camera / Pipeline
# ──────────────────────────────────────────────
@dataclass
class CameraConfig:
    device_index: int = 0
    capture_width: int = 1280
    capture_height: int = 720
    processing_width: int = 640      # internal resize for speed
    processing_height: int = 360
    target_fps: int = 30
    buffer_size: int = 4             # OpenCV internal buffer (low = less latency)


# ──────────────────────────────────────────────
# Head Pose Thresholds  (degrees)
# ──────────────────────────────────────────────
@dataclass
class HeadPoseConfig:
    # Yaw: left/right rotation
    yaw_alert_threshold: float = 25.0      # beyond this → looking away
    yaw_warning_threshold: float = 15.0    # mild deviation

    # Pitch: up/down rotation (Adjusted for recent pose coordinates)
    pitch_down_threshold: float = -150.0
    pitch_up_threshold: float = -210.0

    # Roll: head tilt (less critical)
    roll_alert_threshold: float = 30.0

    # Temporal smoothing via exponential moving average
    ema_alpha: float = 0.35                # higher = faster response, more jitter


# ──────────────────────────────────────────────
# Frame Counter Config (temporal state machine)
# Each behavior has its own increment/decay budget
# ──────────────────────────────────────────────
@dataclass
class FrameCounterConfig:
    # Looking away (yaw)
    look_away_increment: int = 2
    look_away_decay: int = 1
    look_away_trigger: int = 45        # frames above threshold → confirmed event

    # Looking down (pitch)
    look_down_increment: int = 2
    look_down_decay: int = 1
    look_down_trigger: int = 48

    # Face absent
    face_absent_increment: int = 3
    face_absent_decay: int = 2
    face_absent_trigger: int = 35      # faster trigger for face disappearance
    face_absent_timeout_sec: float = 3.0   # seconds before hard alert

    # Phone detected
    phone_increment: int = 4
    phone_decay: int = 2
    phone_trigger: int = 20            # phones are high confidence → faster trigger

    # Different Person proxy detection
    different_person_increment: int = 3
    different_person_decay: int = 1
    different_person_trigger: int = 12

    # Max cap for all counters (prevent runaway accumulation)
    max_counter_value: int = 60


# ──────────────────────────────────────────────
# Suspicion Scoring System
# ──────────────────────────────────────────────
@dataclass
class SuspicionConfig:
    # Per-event score weights
    weight_look_away: float = 0.8
    weight_look_down: float = 1.5
    weight_face_absent: float = 3.0
    weight_phone_detected: float = 3.0
    weight_multiple_faces: float = 4.0
    weight_different_person: float = 5.0
    weight_rapid_movement: float = 0.5

    # Score dynamics
    score_decay_rate: float = 0.88     # multiplied every frame (exponential decay)
    alert_threshold: float = 16.0      # crossing this triggers a cheating event
    warning_threshold: float = 8.0     # visual warning only
    max_score: float = 30.0            # cap to prevent overflow

    # Cooldown after alert (seconds) — prevents log spam
    alert_cooldown_sec: float = 18.0


# ──────────────────────────────────────────────
# YOLO Phone Detection
# ──────────────────────────────────────────────
@dataclass
class YOLOConfig:
    enabled: bool = True
    model_name: str = "best.pt"        # Matched to exam_proctoring_system.py
    confidence_threshold: float = 0.45    # Lowered from 0.6 to match standalone
    iou_threshold: float = 0.5
    input_size: int = 640
    device: str = "auto"
    phone_class_id: int = 67

    min_phone_area_ratio: float = 0.003   # Matched from 0.002
    max_phone_area_ratio: float = 0.35    # Matched from 0.2
    phone_face_proximity_factor: float = 0.55

    # 🔴 NOVEL FILTERS (Relaxed to avoid blocking true positives)
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 4.0

    min_width_px: int = 40
    min_height_px: int = 40

    # 🔴 CLASSIFIER
    use_classifier: bool = False
    classifier_path: str = "phone_classifier.pth"
# ──────────────────────────────────────────────
# MediaPipe Face Mesh
# ──────────────────────────────────────────────
@dataclass
class FaceMeshConfig:
    max_num_faces: int = 2              # detect up to 2 → multi-face alert
    refine_landmarks: bool = True       # enables iris landmarks for better gaze
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5

    # Face stability: if centroid moves more than this (fraction of frame), flag rapid movement
    movement_threshold: float = 0.08   # 8% of frame dimension per frame


# ──────────────────────────────────────────────
# Event / Evidence Config
# ──────────────────────────────────────────────
@dataclass
class EventConfig:
    capture_dir: str = "captures"
    log_dir: str = "logs"
    log_file: str = "logs/proctoring.log"

    # Circular video buffer — keeps last N seconds
    video_buffer_seconds: int = 8
    screenshot_on_alert: bool = True
    video_clip_on_alert: bool = True

    # Video output codec and fps
    clip_codec: str = "mp4v"
    clip_fps: int = 15               # saved clip fps (lower = smaller file)

    # Log rate limiting
    log_min_interval_sec: float = 2.0


# ──────────────────────────────────────────────
# Debug / Overlay Display
# ──────────────────────────────────────────────
@dataclass
class DisplayConfig:
    show_landmarks: bool = False        # MediaPipe face mesh skeleton
    show_face_bbox: bool = True
    show_phone_bbox: bool = True
    show_pose_axes: bool = True
    show_score_bar: bool = True
    show_frame_counters: bool = True
    show_fps: bool = True

    # Overlay colors (BGR)
    color_ok: Tuple[int, int, int] = (50, 205, 50)       # green
    color_warning: Tuple[int, int, int] = (0, 165, 255)  # orange
    color_alert: Tuple[int, int, int] = (0, 0, 220)      # red
    color_info: Tuple[int, int, int] = (200, 200, 200)   # light gray
    color_accent: Tuple[int, int, int] = (255, 200, 0)   # gold

    font_scale_large: float = 0.75
    font_scale_small: float = 0.55
    font_thickness: int = 2

    # Score bar geometry
    score_bar_x: int = 20
    score_bar_y: int = 60
    score_bar_width: int = 220
    score_bar_height: int = 18

# ──────────────────────────────────────────────
# Identity & Tracking Config
# ──────────────────────────────────────────────
@dataclass
class FaceIdentityConfig:
    enabled: bool = True
    model_name: str = "Facenet"
    similarity_threshold: float = 0.40   # Threshold for cosine distance
    verify_every_n_frames: int = 30      # HUGE SPEEDUP: Verify once per ~1 second instead of 5 times
    reference_collection_frames: int = 3 # Build reference significantly faster (was 30)

@dataclass
class TrackingConfig:
    enabled: bool = True
    max_disappeared: int = 30            # Frames before an ID is formally lost
    max_distance: float = 100.0          # Max pixel distance for associating ID



# ──────────────────────────────────────────────
# Master config container
# ──────────────────────────────────────────────
@dataclass
class SystemConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    frame_counters: FrameCounterConfig = field(default_factory=FrameCounterConfig)
    suspicion: SuspicionConfig = field(default_factory=SuspicionConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    face_mesh: FaceMeshConfig = field(default_factory=FaceMeshConfig)
    events: EventConfig = field(default_factory=EventConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    identity: FaceIdentityConfig = field(default_factory=FaceIdentityConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def ensure_dirs(self):
        """Create output directories if missing."""
        os.makedirs(self.events.capture_dir, exist_ok=True)
        os.makedirs(self.events.log_dir, exist_ok=True)


# Module-level singleton — import this everywhere
CFG = SystemConfig()
AppSettings = SystemConfig
