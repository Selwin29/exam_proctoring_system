import cv2
import time
import sys
import os
import argparse
import queue
import numpy as np

# ── Make package imports work when run from multithreaded_pipeline dict ──
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from settings import CFG
from face_monitor import FaceMonitor
from head_pose import HeadPoseEstimator
from phone_detector import PhoneDetector
from handler import EventHandler
from temporal import TemporalStateManager
from scoring import SuspicionScorer
from logger import RateLimitedLogger
from performance import FPSTracker, StageTimer

# Import our new multithreaded components
from camera import CameraThread
from detector import DetectorThread
from renderer import PipelineRenderer

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

    return tsm


def update_score(scorer: SuspicionScorer, tsm: TemporalStateManager, face_state, pose, phone_result):
    sc = CFG.suspicion
    scorer.apply_decay()

    scorer.add_signal("look_away",    sc.weight_look_away, tsm.is_triggered("look_away"))
    scorer.add_signal("look_down",    sc.weight_look_down, tsm.is_triggered("look_down"))
    scorer.add_signal("face_absent",  sc.weight_face_absent, tsm.is_triggered("face_absent"))
    scorer.add_signal("phone",        sc.weight_phone_detected, tsm.is_triggered("phone"))
    scorer.add_signal("multiple_faces", sc.weight_multiple_faces, face_state.multiple_faces)

    rapid_movement = (
        face_state.detected and
        face_state.centroid_velocity > (CFG.face_mesh.movement_threshold * 640) # assumes 640w
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


def run_proctoring_multithreaded(args):
    # Apply args
    CFG.camera.device_index = args.camera
    if args.model: CFG.yolo.model_name = args.model
    if args.threshold is not None: CFG.suspicion.alert_threshold = args.threshold
    if args.no_hud:
        CFG.display.show_score_bar = False
        CFG.display.show_frame_counters = False
        CFG.display.show_fps = False

    CFG.ensure_dirs()

    # Define queues
    frame_queue = queue.Queue(maxsize=1)
    display_queue = queue.Queue(maxsize=1)

    proc_w = CFG.camera.processing_width
    proc_h = CFG.camera.processing_height

    print("[Init] Loading ML Models in main thread...")
    face_monitor = FaceMonitor(CFG.face_mesh, proc_h, proc_w)
    pose_estimator = HeadPoseEstimator(CFG.head_pose, proc_h, proc_w)
    phone_detector = PhoneDetector(CFG.yolo, proc_h, proc_w)

    print("[Init] Starting camera thread...")
    camera_thread = CameraThread(
        device_index=CFG.camera.device_index,
        capture_res=(CFG.camera.capture_width, CFG.camera.capture_height),
        proc_res=(proc_w, proc_h),
        fps=CFG.camera.target_fps,
        buffer_size=CFG.camera.buffer_size,
        out_queue=frame_queue
    )

    print("[Init] Starting detector thread...")
    # Throttled YOLO (e.g. running 1 in every 3 frames)
    yolo_every_n_frames = 3 
    detector_thread = DetectorThread(
        in_queue=frame_queue,
        out_queue=display_queue,
        face_monitor=face_monitor,
        pose_estimator=pose_estimator,
        phone_detector=phone_detector,
        yolo_every_n_frames=yolo_every_n_frames
    )

    # Initialize State Managers
    tsm = _build_temporal_manager()
    scorer = SuspicionScorer(CFG.suspicion)
    logger = RateLimitedLogger("proctoring", CFG.events.log_file, min_interval_sec=CFG.events.log_min_interval_sec)
    event_handler = EventHandler(CFG.events, fps=CFG.events.clip_fps)
    
    fps_tracker = FPSTracker(window_size=30)
    stage_timer = StageTimer()  # We can still track display/sync timing
    renderer = PipelineRenderer(CFG)

    # Start the background threads
    camera_thread.start()
    detector_thread.start()

    print("[Ready] Multithreaded session started. Press 'q' to quit, 'r' to reset, 's' for screenshot.")
    logger.info("session_start", "MT Proctoring session started.", force=True)

    try:
        while True:
            # 1. Pop from detector queue (blocks until a processed frame is ready)
            # This ensures that display FPS is identical to detection FPS, but camera feed won't back up
            try:
                result_data = display_queue.get(timeout=0.1)
                frame_count, raw_frame, proc_frame, face_state, pose, phone_result, detector_time = result_data
            except queue.Empty:
                # Keep main thread responsive to keyboard exits
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            fps_tracker.tick()
            
            # Pass over detector time to stage_timer for hud report compatibility
            stage_timer.history["detector"] = [detector_time] # spoof internal track array

            # 2. Temporal / Scoring Update
            with stage_timer.measure("score"):
                looking_away = pose.looking_left or pose.looking_right
                tsm.update("look_away",   looking_away)
                tsm.update("look_down",   pose.looking_down)
                tsm.update("face_absent", not face_state.detected)
                tsm.update("phone",       phone_result.phone_detected)

                is_alert, is_warning = update_score(scorer, tsm, face_state, pose, phone_result)

            # 3. Alert / Event Handling
            if is_alert:
                handle_event(event_handler, logger, scorer, face_state, pose, phone_result, tsm, proc_frame)
            
            # Logging
            if looking_away:
                logger.warning("look_away", f"Looking {'left' if pose.looking_left else 'right'}", data={"yaw": round(pose.yaw, 1)})
            if pose.looking_down:
                logger.warning("look_down", "Looking down", data={"pitch": round(pose.pitch, 1)})
            if not face_state.detected:
                logger.warning("face_absent", f"Face absent for {face_state.face_absent_seconds:.1f}s")
            if phone_result.phone_detected:
                best = phone_result.best
                logger.warning("phone_detected", "Phone detected", data={"conf": round(best.confidence, 2)})

            # 4. Save frame for event memory buffer
            event_handler.push_frame(proc_frame)

            # 5. Render HUD and Boxes
            with stage_timer.measure("overlay"):
                display_frame = renderer.render(
                    proc_frame=proc_frame,
                    face_state=face_state,
                    pose=pose,
                    phone_result=phone_result,
                    scorer=scorer,
                    tsm=tsm,
                    fps_tracker=fps_tracker,
                    event_count=event_handler.event_count,
                    stage_report=stage_timer.report(),
                    face_monitor=face_monitor,
                    pose_estimator=pose_estimator,
                    phone_detector=phone_detector
                )

            # 6. Display to Screen
            cv2.imshow("Agamotto (MT)", display_frame)

            # 7. Keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[Exit] 'q' pressed. Shutting down variables...")
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
        print("[Exit] Stopping threads...")
        camera_thread.stop()
        detector_thread.stop()
        
        camera_thread.join(timeout=2.0)
        detector_thread.join(timeout=2.0)
        
        face_monitor.release()
        cv2.destroyAllWindows()
        logger.info("session_end", "MT Proctoring session ended.", force=True)
        print(f"\n[Summary] Total events: {event_handler.event_count}")
        print(f"[Summary] Logs: {CFG.events.log_file}")
        print(f"[Summary] Captures: {CFG.events.capture_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description="Multithreaded Agamotto")
    parser.add_argument("--camera",    type=int,   default=0,    help="Camera device index")
    parser.add_argument("--model",     type=str,   default=None, help="YOLO model override (e.g. yolov8l.pt)")
    parser.add_argument("--no-yolo",   action="store_true",      help="Disable phone detection")
    parser.add_argument("--no-hud",    action="store_true",      help="Disable overlay HUD")
    parser.add_argument("--threshold", type=float, default=None, help="Alert threshold override")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_proctoring_multithreaded(args)
