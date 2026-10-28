# Agamotto

A production-level AI-powered Agamotto built with Python, OpenCV, MediaPipe, and YOLOv8.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         VIDEO PIPELINE                          │
│                                                                 │
│  Camera → Resize → RGB Convert                                  │
│      │                                                          │
│      ├──► MediaPipe Face Mesh ──► Face State                   │
│      │         (468 landmarks)         │                        │
│      │                                 ├──► Head Pose (solvePnP)│
│      │                                 │         │              │
│      └──► YOLOv8 (throttled) ──────► Phone │     │              │
│               (COCO class 67)    Detections│     │              │
│                                            │     │              │
│                                   ┌────────▼─────▼─────────┐   │
│                                   │  Temporal State Manager │   │
│                                   │  (Leaky Integrators)    │   │
│                                   │  look_away, look_down,  │   │
│                                   │  face_absent, phone     │   │
│                                   └────────────┬────────────┘   │
│                                                │                │
│                                   ┌────────────▼────────────┐   │
│                                   │   Suspicion Scorer      │   │
│                                   │   Weighted + Decay      │   │
│                                   └────────────┬────────────┘   │
│                                                │                │
│                               ┌────────────────▼─────────────┐ │
│                               │     Alert Threshold Crossed?  │ │
│                               │     Screenshot + Video Clip   │ │
│                               │     Structured Log Entry      │ │
│                               └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Temporal Smoothing via Leaky Integrators

Raw computer vision signals are noisy. A single-frame detection of "looking left" may be a tracking artifact, not actual cheating. Instead of binary on/off detection, each behavioral signal drives a **leaky integrator counter**:

```
value[t+1] = clamp(value[t] + increment, 0, max)   if condition True
value[t+1] = clamp(value[t] - decay,    0, max)    if condition False
triggered  = value >= threshold
```

This means:
- Brief noise spikes decay away without triggering
- Sustained behavior accumulates to trigger threshold
- System recovers naturally when behavior stops

### 2. Suspicion Scoring with Exponential Decay

The scoring system accumulates weighted signals and decays each frame:
```
score[t+1] = score[t] * decay_rate   (decay, ~0.92/frame)
score[t+1] += weight_signal          (if counter triggered)
alert       = score >= alert_threshold
```

This prevents a single phone detection from permanently alerting — the score decays to baseline if the phone disappears.

### 3. Phone Detection False Positive Suppression

YOLOv8 detects phones well, but in exam settings there are many false positives (monitors, picture frames, books). We filter with:
1. **Confidence threshold** (0.45) — removes low-confidence hits
2. **Area ratio filter** — ignores too-small (noise) and too-large (monitor) detections
3. **Spatial proximity to face** — only flags phones within 55% of frame diagonal from face center

### 4. solvePnP Head Pose

MediaPipe landmarks are 2D. We use OpenCV `solvePnP` with a canonical 3D face model to recover true 3D head orientation (yaw/pitch/roll in degrees). EMA smoothing (α=0.35) removes jitter without significant lag.

---

## Project Structure

```
exam_proctoring/
├── exam_proctoring_system.py    ← SINGLE-FILE complete implementation
├── main.py                      ← Modular entry point
├── requirements.txt
│
├── config/
│   ├── __init__.py
│   └── settings.py              ← All configuration parameters
│
├── detection/
│   ├── __init__.py
│   ├── face_monitor.py          ← MediaPipe face mesh + state tracking
│   ├── head_pose.py             ← solvePnP head pose estimation
│   └── phone_detector.py        ← YOLOv8 with spatial filtering
│
├── events/
│   ├── __init__.py
│   └── handler.py               ← Evidence capture (screenshot + video)
│
└── utils/
    ├── __init__.py
    ├── temporal.py              ← Leaky integrator frame counters
    ├── scoring.py               ← Suspicion scoring engine
    ├── logger.py                ← Rate-limited structured logger
    ├── performance.py           ← FPS tracker, stage timer
    └── overlay.py               ← HUD renderer
```

---

## Installation

```bash
# 1. Clone / extract project
cd exam_proctoring

# 2. Install dependencies
pip install -r requirements.txt

# 3. YOLO model auto-downloads on first run (yolov8m.pt ~50MB)

# Optional: GPU acceleration (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

### Single-file (easiest)
```bash
python exam_proctoring_system.py
python exam_proctoring_system.py --camera 1        # different camera
python exam_proctoring_system.py --model yolov8l.pt  # more accurate YOLO
python exam_proctoring_system.py --no-yolo           # disable phone detection
python exam_proctoring_system.py --threshold 8.0     # lower alert sensitivity
```

### Modular version
```bash
python main.py
python main.py --camera 0 --model yolov8l.pt
```

### Runtime Controls
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Manual screenshot |
| `r` | Reset score and all counters |
| `d` | Toggle face mesh landmark overlay |

---

## Configuration Tuning

All parameters are in `config/settings.py` (modular) or the `Config` dataclass in the single file.

### Sensitivity Tuning

| Parameter | Effect | Tune When... |
|-----------|--------|--------------|
| `alert_threshold` | Score required to trigger alert | Too many false alerts → increase |
| `yaw_alert_deg` | Head turn angle for "looking away" | Candidate has a wide desk → increase |
| `pitch_down_deg` | Downward pitch for "looking down" | False triggers from reading → decrease |
| `yolo_conf` | Phone detection confidence | Too many phone false positives → increase |
| `phone_proximity_factor` | How close phone must be to face | Detecting background phones → decrease |
| `score_decay_rate` | How fast score falls | Score stays high too long → decrease |

### Performance Tuning

| Parameter | Effect |
|-----------|--------|
| `yolo_every_n_frames` | Run YOLO every N frames (higher = faster, less responsive) |
| `proc_width/height` | Internal processing resolution (lower = faster) |
| `yolo_imgsz` | YOLO input size (320 = fast, 640 = accurate) |
| `yolo_model` | `yolov8n.pt` (fastest) → `yolov8l.pt` (most accurate) |

---

## Evidence Output

On each alert:
- **Screenshot**: `captures/EVT_<timestamp>_shot_<datetime>.jpg`
  - Includes red alert banner with reason and score
- **Video clip**: `captures/EVT_<timestamp>_clip_<datetime>.mp4`
  - Last 8 seconds of video before/during alert
- **Log entry**: `logs/proctoring.log`
  - JSON-structured with timestamp, event type, score, signal details

---

## HUD Overlay Guide

```
┌──────────────────────────────────────────────────────────────┐
│ [LEFT PANEL - semi-transparent]                              │
│  FPS: 28.3                                                   │
│                                                              │
│  SUSPICION  7.2/10.0    [====------]  WARNING                │
│                                                              │
│  Look Away  (18)  [=======---]                               │
│  Look Down  (0)   [----------]                               │
│  No Face    (0)   [----------]                               │
│  Phone      (0)   [----------]                               │
│                                                              │
│  HEAD POSE                                                   │
│    Yaw:  +28.3°   ← highlighted orange (above threshold)    │
│    Pitch: -8.1°                                              │
│    Roll:  +2.4°                                              │
│                                                              │
│  FACE                                                        │
│    1 face detected                                           │
│                                                              │
│  PHONE                                                       │
│    Not detected                                              │
│                                                              │
│  Events: 2                                                   │
│  face=1ms | pose=0ms | yolo=28ms | overlay=3ms               │
└──────────────────────────────────────────────────────────────┘
                    ┌──────────────────────────┐ [TOP-CENTER BANNER]
                    │  SUSPICIOUS BEHAVIOR...  │
                    └──────────────────────────┘
                                               ┌───────────┐
                                               │ ◄ LEFT    │ [TOP-RIGHT PILLS]
                                               └───────────┘
```

---

## Extending the System

### Add a new behavioral signal

1. Register a counter in `TemporalStateManager`:
```python
tsm.register("eyes_closed", cfg.counter_eyes_closed)
```

2. Detect the condition from face landmarks:
```python
eyes_closed = detect_eye_closure(face_state.landmarks)
tsm.update("eyes_closed", eyes_closed)
```

3. Add to scorer:
```python
scorer.add("eyes_closed", cfg.weight_eyes_closed, tsm.is_triggered("eyes_closed"))
```

4. Add to event reason builder in `handle_event()`.

### Add a new detection model

Create a new file in `detection/` following the same pattern as `phone_detector.py`:
- Accept `cfg` and `frame_h/w` in `__init__`
- Return a typed result dataclass from `process()`
- Include a `draw_*()` overlay method

---

## Performance Benchmarks (approximate)

| Component | CPU Time (i7, no GPU) |
|-----------|----------------------|
| Frame resize | ~0.5ms |
| MediaPipe Face Mesh | ~12-18ms |
| solvePnP pose | ~0.1ms |
| YOLOv8m (CPU) | ~80-150ms (throttled ÷3) |
| YOLOv8m (CUDA) | ~8-15ms |
| HUD overlay | ~2-4ms |
| **Total @ 640×360** | **~25-35ms → 30 FPS** (GPU) / **~20-25 FPS** (CPU) |

For CPU-only systems, use `yolov8n.pt` and increase `yolo_every_n_frames` to 5.
