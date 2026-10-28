"""
main.py
───────
Agamotto — Main Pipeline Orchestrator
Optimized for Real-Time Performance using Threading!
"""

import cv2
import time
import sys
import os
import argparse
import queue
import threading
import numpy as np

# ── Make package imports work when run from project root ────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings import CFG
from face_monitor import FaceMonitor
from head_pose import HeadPoseEstimator
from phone_detector import PhoneDetector
from handler import EventHandler
from temporal import TemporalStateManager
from scoring import SuspicionScorer
from logger import RateLimitedLogger
from performance import FPSTracker, StageTimer
from overlay import HUDRenderer
from tracking import FaceTracker
from face_identity import FaceIdentityVerifier

# ────────────────────────────────────────────────────────────────────────────
# Initialization helpers
# ────────────────────────────────────────────────────────────────────────────

def _build_temporal_manager() -> TemporalStateManager:
    """Register all behavioral counters with their tuned increment/decay/trigger values."""
    tsm = TemporalStateManager()
    fc = CFG.frame_counters

    tsm.register("look_away",   trigger=fc.look_away_trigger,
                 increment=fc.look_away_increment, decay=fc.look_away_decay,
                 max_value=fc.max_counter_value)
    tsm.register("look_down",   trigger=fc.look_down_trigger,
                 increment=fc.look_down_increment, decay=fc.look_down_decay,
                 max_value=fc.max_counter_value)
    tsm.register("face_absent", trigger=fc.face_absent_trigger,
                 increment=fc.face_absent_increment, decay=fc.face_absent_decay,
                 max_value=fc.max_counter_value)
    tsm.register("phone",       trigger=fc.phone_trigger,
                 increment=fc.phone_increment, decay=fc.phone_decay,
                 max_value=fc.max_counter_value)

    if hasattr(fc, "different_person_trigger"):
        tsm.register("different_person", trigger=fc.different_person_trigger,
                     increment=fc.different_person_increment, decay=fc.different_person_decay,
                     max_value=fc.max_counter_value)

    return tsm


def _open_camera(cfg) -> cv2.VideoCapture:
    """Open the camera with preferred resolution and buffer settings."""
    cap = cv2.VideoCapture(cfg.device_index)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {cfg.device_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.capture_height)
    cap.set(cv2.CAP_PROP_FPS,          cfg.target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   cfg.buffer_size)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] Opened at {actual_w}x{actual_h}")

    return cap, actual_w, actual_h


def update_score(scorer: SuspicionScorer, tsm: TemporalStateManager, face_state, pose, phone_result):
    sc = CFG.suspicion
    scorer.apply_decay()

    scorer.add_signal("look_away",    sc.weight_look_away, tsm.is_triggered("look_away"))
    scorer.add_signal("look_down",    sc.weight_look_down, tsm.is_triggered("look_down"))
    scorer.add_signal("face_absent",  sc.weight_face_absent, tsm.is_triggered("face_absent"))
    scorer.add_signal("phone",        sc.weight_phone_detected, tsm.is_triggered("phone"))
    scorer.add_signal("multiple_faces", sc.weight_multiple_faces, face_state.multiple_faces)

    if hasattr(sc, "weight_different_person"):
        scorer.add_signal("different_person", sc.weight_different_person, tsm.is_triggered("different_person"))

    rapid_movement = (
        face_state.detected and
        face_state.centroid_velocity > (CFG.face_mesh.movement_threshold * 640)
    )
    scorer.add_signal("rapid_movement", sc.weight_rapid_movement, rapid_movement)
    return scorer.evaluate()


def handle_event(event_handler: EventHandler, logger: RateLimitedLogger, scorer: SuspicionScorer,
                 face_state, pose, phone_result, tsm: TemporalStateManager, current_frame: np.ndarray):
    active = scorer.active_signals
    reasons = []

    if "phone" in active: reasons.append("PHONE DETECTED")
    if "face_absent" in active: reasons.append(f"FACE ABSENT {face_state.face_absent_seconds:.1f}s")
    if "look_away" in active:
        direction = "LEFT" if pose.looking_left else "RIGHT"
        reasons.append(f"LOOKING {direction} ({pose.yaw:+.0f}°)")
    if "look_down" in active: reasons.append(f"LOOKING DOWN ({pose.pitch:+.0f}°)")
    if "multiple_faces" in active: reasons.append(f"MULTIPLE FACES ({face_state.num_faces})")
    if "rapid_movement" in active: reasons.append("RAPID HEAD MOVEMENT")
    if "different_person" in active: reasons.append("DIFFERENT PERSON DETECTED (PROXY)")

    reason_str = " | ".join(reasons) if reasons else "Sustained suspicious behavior"

    event = event_handler.handle_alert(
        reason=reason_str,
        suspicion_score=scorer.score,
        active_signals=active,
        current_frame=current_frame,
    )

    if event:
        logger.alert("cheating_event", reason_str, data={
            "event_id": event.event_id,
            "score": scorer.score,
            "screenshot": event.screenshot_path,
            "clip": event.video_clip_path,
        })
        print(f"\n[ALERT] {reason_str}  score={scorer.score:.1f}  id={event.event_id}")

    return event

# ────────────────────────────────────────────────────────────────────────────
# THREAD 1: CAMERA WORKER
# ────────────────────────────────────────────────────────────────────────────
class CameraThread(threading.Thread):
    def __init__(self, cap, proc_w, proc_h, out_queue):
        """Continuously pulls frames without blocking the main event loops."""
        super().__init__(daemon=True)
        self.cap = cap
        self.proc_w = proc_w
        self.proc_h = proc_h
        self.out_queue = out_queue
        self.running = False
        self.frame_index = 0

    def run(self):
        self.running = True
        while self.running:
            ret, raw_frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            
            self.frame_index += 1
            # Resize inline to assist the compute bottleneck
            proc_frame = cv2.resize(raw_frame, (self.proc_w, self.proc_h))
            rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            
            data = (self.frame_index, raw_frame, proc_frame, rgb_frame)
            try:
                self.out_queue.put_nowait(data)
            except queue.Full:
                try:
                    self.out_queue.get_nowait() # pop oldest frame to maintain low latency
                    self.out_queue.put_nowait(data)
                except queue.Empty:
                    pass

    def stop(self):
        self.running = False


# ────────────────────────────────────────────────────────────────────────────
# THREAD 2: DETECTION WORKER
# ────────────────────────────────────────────────────────────────────────────
class DetectorThread(threading.Thread):
    def __init__(self, face_monitor, pose_estimator, phone_detector, face_tracker, face_verifier, in_queue, out_queue, yolo_every_n=3):
        """Processes logic concurrently. MediaPipe runs per frame, YOLO/DeepFace throttled."""
        super().__init__(daemon=True)
        self.face_monitor = face_monitor
        self.pose_estimator = pose_estimator
        self.phone_detector = phone_detector
        self.face_tracker = face_tracker
        self.face_verifier = face_verifier
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.yolo_every_n = yolo_every_n
        self.running = False

    def run(self):
        self.running = True
        from phone_detector import PhoneDetectionResult
        last_phone_result = PhoneDetectionResult()
        
        while self.running:
            try:
                try:
                    data = self.in_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                    
                frame_index, raw_frame, proc_frame, rgb_frame = data
                
                start_t = time.time()
                
                # --- 1) MediaPipe Face Mesh ---
                face_state = self.face_monitor.process(rgb_frame)
                
                # --- 2) Head Pose Estimation ---
                pose = self.pose_estimator.estimate(face_state)
                
                # --- Tracking (Every frame) ---
                tracked_objects, proxy_alert_tracker = self.face_tracker.process_faces(face_state, self.face_monitor.frame_w, self.face_monitor.frame_h)
                
                # --- Identity Verification (Throttled offset) ---
                identity_alert = False
                # We offset the identity verifier frame by 1 to not run concurrently with YOLO on the same cycle
                if frame_index % self.face_verifier.cfg.verify_every_n_frames == 1:
                    identity_alert = self.face_verifier.process(rgb_frame, face_state.bbox_norm)
                
                is_different_person = proxy_alert_tracker or identity_alert

                # --- 3) YOLOv8 Phone Detector (Throttled & Cached) ---
                if frame_index % self.yolo_every_n == 0:
                    last_phone_result = self.phone_detector.process(proc_frame, face_state.centroid_px)
                
                detector_t = time.time() - start_t
                
                # Push strictly non-blocking
                out_data = (frame_index, raw_frame, proc_frame, face_state, pose, tracked_objects, is_different_person, last_phone_result, detector_t)
                try:
                    self.out_queue.put_nowait(out_data)
                except queue.Full:
                    try:
                        self.out_queue.get_nowait()
                        self.out_queue.put_nowait(out_data)
                    except queue.Empty:
                        pass
            except Exception as e:
                import traceback
                print(f"[DetectorThread Error] {e}")
                traceback.print_exc()
                import sys; sys.stdout.flush()

    def stop(self):
        self.running = False


# ────────────────────────────────────────────────────────────────────────────
# THREAD 3: MAIN ORCHESTRATOR / RENDERER LOOP
# ────────────────────────────────────────────────────────────────────────────
def run_proctoring(args, reference_embedding=None):
    CFG.ensure_dirs()

    # ── Init Camera & Resolutons ───────────────────────────────────────────────
    cap, cam_w, cam_h = _open_camera(CFG.camera)
    proc_w = CFG.camera.processing_width
    proc_h = CFG.camera.processing_height

    # SPLASH SCREEN / CAMERA WARMUP
    splash_frame = None
    for _ in range(20):
        ret, frame = cap.read()
        if ret and frame is not None:
            splash_frame = frame
            break
        time.sleep(0.1)

    if splash_frame is not None:
        splash_frame = cv2.resize(splash_frame, (proc_w, proc_h))
        cv2.putText(splash_frame, "INITIALIZING MODELS...", (50, proc_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
        cv2.putText(splash_frame, "Please wait, this may take a moment.", (50, proc_h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        window_name = "Agamotto"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow(window_name, splash_frame)
        for _ in range(10):
            cv2.waitKey(20)
    else:
        print("[Warning] Camera failed to warm up for splash screen.")

    # ── Init Models ─────────────────────────────────────────────────────────
    print("[Init] Loading MediaPipe Face Mesh...")
    face_monitor = FaceMonitor(CFG.face_mesh, proc_h, proc_w)

    print("[Init] Initializing head pose estimator...")
    pose_estimator = HeadPoseEstimator(CFG.head_pose, proc_h, proc_w)

    print("[Init] Preparing YOLOv8 phone detector (lazy load)...")
    phone_detector = PhoneDetector(CFG.yolo, proc_h, proc_w)

    print("[Init] Initializing trackers & identity verifier...")
    face_tracker = FaceTracker(CFG.tracking)
    face_verifier = FaceIdentityVerifier(CFG.identity)
    if reference_embedding is not None:
        face_verifier.set_reference_embedding(reference_embedding)

    # ── Score Managers & System Output ─────────────────────────────────────────
    tsm = _build_temporal_manager()
    scorer = SuspicionScorer(CFG.suspicion)
    logger = RateLimitedLogger("proctoring", CFG.events.log_file, min_interval_sec=CFG.events.log_min_interval_sec)
    event_handler = EventHandler(CFG.events, fps=CFG.events.clip_fps)
    hud = HUDRenderer(CFG.display, CFG.suspicion)
    
    fps_tracker = FPSTracker(window_size=30)
    stage_timer = StageTimer()

    # ── Thread Queues ───────────────────────────────────────────────────────
    # Size 1 guarantees minimal latency (we only ever care about newest frame)
    frame_queue = queue.Queue(maxsize=1)   
    display_queue = queue.Queue(maxsize=1) 

    # ── Launch Background Tasks ───────────────────────────────────────────────
    camera_thread = CameraThread(cap, proc_w, proc_h, frame_queue)
    detector_thread = DetectorThread(face_monitor, pose_estimator, phone_detector, face_tracker, face_verifier, frame_queue, display_queue, yolo_every_n=3)

    camera_thread.start()
    detector_thread.start()

    print("[Ready] Multithreaded session active. Press 'q' to quit, 's' for screenshot.\n")
    logger.info("session_start", "MT Proctoring session started.", force=True)

    try:
        while True:
            # ── 1. Render/Main Context Retrieves Frame ───────────────────────
            try:
                # Polling enables keyboard interrupt stability
                result = display_queue.get(timeout=0.1)
                frame_index, raw_frame, proc_frame, face_state, pose, tracked_objects, is_different_person, phone_result, detector_t = result
            except queue.Empty:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            fps_tracker.tick()
            
            # Map background processing time properly for display metrics
            stage_timer._times["multi_thread"] = detector_t * 1000

            # ── 2. Run Temporal State Update + Logic ─────────────────────────
            with stage_timer.measure("score_event"):
                looking_away = pose.looking_left or pose.looking_right
                tsm.update("look_away",   looking_away)
                tsm.update("look_down",   pose.looking_down)
                tsm.update("face_absent", not face_state.detected)
                tsm.update("phone",       phone_result.phone_detected)
                if hasattr(CFG.frame_counters, "different_person_trigger"):
                    tsm.update("different_person", is_different_person)

                is_alert, is_warning = update_score(scorer, tsm, face_state, pose, phone_result)

                if is_alert:
                    handle_event(event_handler, logger, scorer, face_state, pose, phone_result, tsm, proc_frame)

                # Limited logging logic natively maintained
                if looking_away:
                    logger.warning("look_away", f"Looking {'left' if pose.looking_left else 'right'}", data={"yaw": round(pose.yaw, 1)})
                if pose.looking_down:
                    logger.warning("look_down", "Looking down", data={"pitch": round(pose.pitch, 1)})
                if not face_state.detected:
                    logger.warning("face_absent", f"Face absent for {face_state.face_absent_seconds:.1f}s")
                if phone_result.phone_detected:
                    best = phone_result.best
                    logger.warning("phone_detected", "Phone detected", data={"conf": round(best.confidence, 2)})

            # Keep video history accurate
            event_handler.push_frame(proc_frame)

            # ── 3. Render and Draw using Existing HUD logic ──────────────────
            with stage_timer.measure("overlay"):
                display_frame = proc_frame.copy()

                # Draw tracked IDs
                if face_state.detected and tracked_objects is not None:
                    for obj_id, centroid in tracked_objects.items():
                        text = f"ID: {obj_id}"
                        cv2.putText(display_frame, text, (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if CFG.display.show_face_bbox:
                    bbox_color = (
                        CFG.display.color_alert if not face_state.detected
                        else (CFG.display.color_warning if is_warning else CFG.display.color_ok)
                    )
                    face_monitor.draw_face_bbox(display_frame, face_state, bbox_color)

                if CFG.display.show_landmarks:
                    face_monitor.draw_landmarks(display_frame, face_state)

                if CFG.display.show_pose_axes and pose.valid:
                    pose_estimator.draw_pose_axes(display_frame, pose)

                if CFG.display.show_phone_bbox:
                    phone_detector.draw_detections(display_frame, phone_result, CFG.display.color_alert)

                hud.render(
                    display_frame,
                    face_state, pose, phone_result,
                    scorer, tsm, fps_tracker,
                    event_handler.event_count,
                    stage_report=stage_timer.report(),
                )

            # ── 4. Actual OpenCV imshow (Render loop) ────────────────────────
            window_name = "Agamotto"
            # Ensure window is created and set to remain always-on-top
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[Exit] 'q' pressed. Shutting down.")
                break
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(CFG.events.capture_dir, f"manual_{ts}.jpg")
                cv2.imwrite(path, display_frame)
                print(f"[Screenshot] Saved: {path}")
            elif key == ord('r'):
                scorer.reset()
                tsm.reset_all()
                print("[Reset] Score and counters reset.")
            elif key == ord('d'):
                CFG.display.show_landmarks = not CFG.display.show_landmarks

    except KeyboardInterrupt:
        print("\n[Exit] Keyboard interrupt.")

    finally:
        print("[Exit] Requesting threads to stop...")
        # Stop flag triggers graceful close
        camera_thread.stop()
        detector_thread.stop()
        
        # Merge background executions back to sync main exit 
        camera_thread.join(timeout=2.0)
        detector_thread.join(timeout=2.0)

        if cap.isOpened(): cap.release()
        face_monitor.release()
        cv2.destroyAllWindows()
        logger.info("session_end", "Proctoring session ended.", force=True)
        print(f"\n[Summary] Total events: {event_handler.event_count}")

# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Agamotto")
    parser.add_argument("--camera",    type=int,   default=0,    help="Camera device index")
    parser.add_argument("--model",     type=str,   default=None, help="YOLO model override (e.g. yolov8l.pt)")
    parser.add_argument("--no-yolo",   action="store_true",      help="Disable phone detection")
    parser.add_argument("--no-hud",    action="store_true",      help="Disable overlay HUD")
    parser.add_argument("--threshold", type=float, default=None, help="Alert threshold override")
    
    # Flask integration args
    parser.add_argument('--user-id',        type=int,   default=0)
    parser.add_argument('--session-id',     type=int,   default=0)
    parser.add_argument('--db-path',        type=str,   default='')
    parser.add_argument('--embedding-file', type=str,   default='')
    
    return parser.parse_args()


if __name__ == "__main__":
    import signal
    import json
    
    def _shutdown(sig, frame):
        evt = {'type': 'proctor_stopped', 'detail': 'Proctoring session ended gracefully', 'severity': 'info'}
        print(json.dumps(evt), flush=True)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)

    args = parse_args()

    CFG.camera.device_index = args.camera
    if args.model: CFG.yolo.model_name = args.model
    if args.no_yolo: CFG.yolo.enabled = False
    if args.threshold is not None: CFG.suspicion.alert_threshold = args.threshold
    if args.no_hud:
        CFG.display.show_score_bar = False
        CFG.display.show_frame_counters = False
        CFG.display.show_fps = False

    reference_embedding = None
    if args.embedding_file:
        try:
            with open(args.embedding_file, 'r') as f:
                data = json.load(f)
            reference_embedding = np.array(data, dtype=np.float32)
        except Exception as e:
            print(json.dumps({'type': 'embedding_load_error', 'detail': str(e), 'severity': 'medium'}), flush=True)

    run_proctoring(args, reference_embedding=reference_embedding)
