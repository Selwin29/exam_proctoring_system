import sys
import os
import cv2

# Add parent directory to path to import core models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overlay import HUDRenderer

class PipelineRenderer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hud = HUDRenderer(cfg.display, cfg.suspicion)
        
    def render(self, proc_frame, face_state, pose, phone_result, scorer, tsm, fps_tracker, event_count, stage_report, face_monitor, pose_estimator, phone_detector):
        """
        Draws all overlays on the processed frame.
        """
        display_frame = proc_frame.copy()

        # Face bounding box
        if self.cfg.display.show_face_bbox:
            bbox_color = (
                self.cfg.display.color_alert if not face_state.detected
                else (self.cfg.display.color_warning if scorer.score >= self.cfg.suspicion.warning_threshold
                      else self.cfg.display.color_ok)
            )
            # Use the core face_monitor to draw
            if hasattr(face_monitor, 'draw_face_bbox'):
                face_monitor.draw_face_bbox(display_frame, face_state, bbox_color)

        # Landmark dots (optional)
        if self.cfg.display.show_landmarks and hasattr(face_monitor, 'draw_landmarks'):
            face_monitor.draw_landmarks(display_frame, face_state)

        # 3D pose axes
        if self.cfg.display.show_pose_axes and pose.valid and hasattr(pose_estimator, 'draw_pose_axes'):
            pose_estimator.draw_pose_axes(display_frame, pose)

        # Phone bounding boxes
        if self.cfg.display.show_phone_bbox and hasattr(phone_detector, 'draw_detections'):
            phone_detector.draw_detections(
                display_frame, phone_result, self.cfg.display.color_alert
            )

        # HUD panel
        self.hud.render(
            display_frame,
            face_state, pose, phone_result,
            scorer, tsm, fps_tracker,
            event_count,
            stage_report=stage_report,
        )
        
        return display_frame
