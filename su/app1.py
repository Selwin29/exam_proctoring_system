"""
app.py — Agamotto Professional Monitoring Dashboard
────────────────────────────────────────────────────
Refactored for performance, reliability, and modern UI integration.

This version implements:
  - Multi-threaded background capture with robust state synchronization.
  - Violation Feed tracking (last 50 events).
  - High-frequency live stats for real-time dashboard updates.
  - Explicit synchronization with calibrated head pose thresholds.
"""

import threading
import time
import json
import cv2
import os
from flask import Flask, Response, jsonify, render_template, send_from_directory
from typing import List, Dict, Optional

# ── Backend module imports ────────────────────────────────────────────────────
from settings import CFG
from face_monitor import FaceMonitor
from head_pose import HeadPoseEstimator
from phone_detector import PhoneDetector
from scoring import SuspicionScorer
from temporal import TemporalStateManager
from logger import RateLimitedLogger
from performance import FPSTracker

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Shared Application State ──────────────────────────────────────────────────
class AppState:
    """Thread-safe container for all live proctoring data."""
    def __init__(self):
        self._lock = threading.Lock()
        self.frame_index = 0
        self.live = {
            "is_active": False,
            "face_present": False,
            "head_direction": "Straight",
            "is_looking_down": False,
            "phone_detected": False,
            "suspicion_score": 0.0,
            "alert_level": "Normal", # Normal | Warning | Alert
            "fps": 0.0,
            "events_count": 0,
            "session_time": 0.0,
        }
        self.violations: List[Dict] = []
        self.latest_frame: bytes = b''
        self.start_ts = time.time()

    def update(self, snapshot: Dict):
        with self._lock:
            self.live.update(snapshot)
            self.live["session_time"] = time.time() - self.start_ts

    def add_violation(self, reason: str, level: str, score: float):
        with self._lock:
            evt = {
                "id": f"V_{int(time.time())}_{len(self.violations)}",
                "time": time.strftime("%H:%M:%S"),
                "reason": reason,
                "level": level,
                "score": round(score, 1)
            }
            self.violations.insert(0, evt) # Prepend newest
            if len(self.violations) > 50:
                self.violations.pop()

    def set_frame(self, frame_bytes: bytes):
        with self._lock:
            self.latest_frame = frame_bytes

    def get_snapshot(self) -> Dict:
        with self._lock:
            return {
                "live": self.live,
                "violations": self.violations[:10] # send last 10
            }

state = AppState()


# ── Background Process ────────────────────────────────────────────────────────
def proctoring_engine(cfg):
    """
    Core detection loop running in a background thread.
    Synchronizes all modules and updates the shared AppState.
    """
    # ── Initialize Subsystems ──────────────────────────────────────────────
    proc_h, proc_w = cfg.camera.processing_height, cfg.camera.processing_width
    
    face_proc = FaceMonitor(cfg.face_mesh, proc_h, proc_w)
    pose_proc = HeadPoseEstimator(cfg.head_pose, proc_h, proc_w)
    yolo_proc = PhoneDetector(cfg.yolo, proc_h, proc_w)
    tsm       = TemporalStateManager()
    scorer    = SuspicionScorer(cfg.suspicion)
    fps       = FPSTracker()
    
    # Register temporal counters
    fc = cfg.frame_counters
    tsm.register("look_away",   fc.look_away_trigger, fc.look_away_increment, fc.look_away_decay, fc.max_counter_value)
    tsm.register("look_down",   fc.look_down_trigger, fc.look_down_increment, fc.look_down_decay, fc.max_counter_value)
    tsm.register("face_absent", fc.face_absent_trigger, fc.face_absent_increment, fc.face_absent_decay, fc.max_counter_value)
    tsm.register("phone",       fc.phone_trigger, fc.phone_increment, fc.phone_decay, fc.max_counter_value)

    # Logging
    logger = RateLimitedLogger("app", cfg.events.log_file, min_interval_sec=1.5)

    # ── Camera Setup ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cfg.camera.device_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.camera.capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.capture_height)
    
    print(f"[Engine] Live monitor started on cam {cfg.camera.device_index}")
    state.update({"is_active": True})

    frame_idx = 0
    last_phone_res = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_idx += 1
            fps.tick()
            
            # 1. Pipeline execution
            proc_frame = cv2.resize(frame, (proc_w, proc_h))
            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            
            face = face_proc.process(rgb)
            pose = pose_proc.estimate(face)
            
            if frame_idx % 3 == 0:
                last_phone_res = yolo_proc.process(proc_frame, face.centroid_px)
            phone = last_phone_res

            # 2. Temporal State
            looking_away = pose.looking_left or pose.looking_right
            tsm.update("look_away",   looking_away)
            tsm.update("look_down",   pose.looking_down)
            tsm.update("face_absent", not face.detected)
            tsm.update("phone",       phone.phone_detected if phone else False)

            # 3. Scoring
            scorer.apply_decay()
            scorer.add_signal("look_away",   cfg.suspicion.weight_look_away,   tsm.is_triggered("look_away"))
            scorer.add_signal("look_down",   cfg.suspicion.weight_look_down,   tsm.is_triggered("look_down"))
            scorer.add_signal("face_absent", cfg.suspicion.weight_face_absent, tsm.is_triggered("face_absent"))
            scorer.add_signal("phone",       cfg.suspicion.weight_phone_detected, tsm.is_triggered("phone"))
            scorer.add_signal("multi_face",  cfg.suspicion.weight_multiple_faces, face.multiple_faces)

            is_alert, is_warning = scorer.evaluate()
            level = "Alert" if is_alert else ("Warning" if is_warning else "Normal")

            # 4. Handle Significant Violations
            should_record = False
            v_reason = ""
            if tsm.is_triggered("look_down"):  v_reason = "Looking Down"
            elif tsm.is_triggered("look_away"): v_reason = "Looking Away"
            elif tsm.is_triggered("phone"):     v_reason = "Phone Detected"
            elif face.multiple_faces:           v_reason = "Multiple Faces"
            
            if v_reason and level != "Normal":
                if logger.event("violation", v_reason):
                    state.add_violation(v_reason, level, scorer.score)

            # 5. Update UI State
            head_dir = "Center"
            if pose.looking_left: head_dir = "Left"
            elif pose.looking_right: head_dir = "Right"
            elif pose.looking_up: head_dir = "Up"

            state.update({
                "face_present":    face.detected,
                "head_direction":  head_dir,
                "is_looking_down": pose.looking_down,
                "phone_detected":  phone.phone_detected if phone else False,
                "suspicion_score": round(scorer.score, 2),
                "alert_level":     level,
                "fps":             round(fps.fps, 1),
            })

            # 6. MJPEG Encoding
            # Minimal HUD drawing for web feed
            cv2.rectangle(proc_frame, (0,0), (proc_w, 30), (0,0,0), -1)
            status_color = (0, 255, 0)
            if is_alert: status_color = (0, 0, 255)
            elif is_warning: status_color = (0, 165, 255)
            
            cv2.putText(proc_frame, f"AGAMOTTO - {level.upper()}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

            if phone and phone.phone_detected:
                for d in phone.detections:
                    bx = d.bbox_px
                    cv2.rectangle(proc_frame, (bx[0], bx[1]), (bx[2], bx[3]), (0,0,255), 2)

            _, buf = cv2.imencode(".jpg", proc_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            state.set_frame(buf.tobytes())

    finally:
        cap.release()
        state.update({"is_active": False})


# ── MJPEG Stream Generator ────────────────────────────────────────────────────
def gen_frames():
    while True:
        frame = state.latest_frame
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)


# ── Flask Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/stats")
def get_stats():
    return jsonify(state.get_snapshot())

@app.route("/captures/<path:filename>")
def get_capture(filename):
    return send_from_directory(CFG.events.capture_dir, filename)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ensure directories exist
    CFG.ensure_dirs()
    
    # Start engine in background
    thread = threading.Thread(target=proctoring_engine, args=(CFG,), daemon=True)
    thread.start()
    
    # Start Web Server
    print("\n" + "="*50)
    print(" Agamotto dashboard: http://0.0.0.0:3000")
    print("="*50 + "\n")
    
    app.run(host="0.0.0.0", port=3000, debug=False, threaded=True)
