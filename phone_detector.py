# detection/phone_detector.py

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

from settings import YOLOConfig

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────

# DATA CLASSES

# ─────────────────────────────────────────────────────────────

@dataclass
class PhoneDetection:
    bbox_px: Tuple[int, int, int, int]
    confidence: float
    area_ratio: float
    distance_to_face: float

@dataclass
class PhoneDetectionResult:
    detections: List[PhoneDetection] = field(default_factory=list)
    raw_count: int = 0
    filtered_count: int = 0

    @property
    def phone_detected(self) -> bool:
        return len(self.detections) > 0

    @property
    def best(self) -> Optional[PhoneDetection]:
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.confidence)

# ─────────────────────────────────────────────────────────────

# MAIN DETECTOR

# ─────────────────────────────────────────────────────────────

class PhoneDetector:

    def __init__(self, cfg: YOLOConfig, frame_h: int, frame_w: int):
        self.cfg = cfg
        self.frame_h = frame_h
        self.frame_w = frame_w
        self._model_custom = None
        self._model_default = None
        self._device = self._resolve_device(cfg.device)
        self._frame_area = frame_h * frame_w

        # 🔥 Classifier (optional)
        self._clf_model = None
        self._clf_transform = None

        if getattr(self.cfg, "use_classifier", False):
            self._load_classifier()

    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        if device_str == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except:
                pass
            return "cpu"
        return device_str

    # ─────────────────────────────────────────────────────────

    def _ensure_model_loaded(self):
        if self._model_custom is not None:
            return

        try:
            from ultralytics import YOLO

            logger.info(f"Loading dual YOLO models on {self._device}")

            self._model_custom = YOLO("best.pt")
            self._model_default = YOLO("yolov8n.pt")
            self._model_custom.to(self._device)
            self._model_default.to(self._device)

            # Warmup
            dummy = np.zeros((self.cfg.input_size, self.cfg.input_size, 3), dtype=np.uint8)

            self._model_custom.predict(
                dummy,
                device=self._device,
                imgsz=self.cfg.input_size,
                verbose=False,
                half=True if self._device == "cuda" else False
            )
            self._model_default.predict(
                dummy,
                device=self._device,
                imgsz=self.cfg.input_size,
                verbose=False,
                half=True if self._device == "cuda" else False
            )

            logger.info("YOLO ready")

        except Exception as e:
            logger.error(f"YOLO load error: {e}")
            self._model_custom = None
            self._model_default = None

    # ─────────────────────────────────────────────────────────
    # 🔥 CLASSIFIER LOADER
    # ─────────────────────────────────────────────────────────

    def _load_classifier(self):
        try:
            import torch
            import torchvision.transforms as transforms

            self._clf_model = torch.load(self.cfg.classifier_path, map_location=self._device)
            self._clf_model.eval()

            self._clf_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            logger.info("Phone classifier loaded")

        except Exception as e:
            logger.warning(f"Classifier load failed: {e}")
            self._clf_model = None

    # ─────────────────────────────────────────────────────────
    # 🔥 CLASSIFIER FUNCTION
    # ─────────────────────────────────────────────────────────

    def _classify_crop(self, frame, bbox):
        if self._clf_model is None:
            return True

        import torch
        from PIL import Image

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return False

        img = Image.fromarray(crop)
        img = self._clf_transform(img).unsqueeze(0)

        with torch.no_grad():
            out = self._clf_model(img)
            pred = torch.argmax(out, dim=1).item()

        return pred == 1  # 1 = phone

    # ─────────────────────────────────────────────────────────
    # MAIN PROCESS
    # ─────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray, face_centroid_px: Optional[Tuple[int, int]]) -> PhoneDetectionResult:
        result = PhoneDetectionResult()
        
        if not self.cfg.enabled:
            return result

        self._ensure_model_loaded()

        if self._model_custom is None:
            return result

        try:
            pred1 = self._model_custom.predict(
                frame,
                conf=self.cfg.confidence_threshold,
                device=self._device,
                imgsz=self.cfg.input_size,
                verbose=False,
                half=True if self._device == "cuda" else False,
                classes=[0]  # Assuming best.pt is single-class phone model
            )
            pred2 = self._model_default.predict(
                frame,
                conf=self.cfg.confidence_threshold,
                device=self._device,
                imgsz=self.cfg.input_size,
                verbose=False,
                half=True if self._device == "cuda" else False,
                classes=[self.cfg.phone_class_id]  # Only detect phones (class 67)
            )
            
            all_boxes = []
            if pred1 and pred1[0].boxes is not None:
                all_boxes.extend(pred1[0].boxes)
            if pred2 and pred2[0].boxes is not None:
                all_boxes.extend(pred2[0].boxes)
                
        except Exception as e:
            logger.warning(f"Inference error: {e}")
            return result

        if not all_boxes:
            return result

        result.raw_count = len(all_boxes)

        h, w = self.frame_h, self.frame_w
        diag = (h**2 + w**2) ** 0.5

        for box in all_boxes:

            conf = float(box.conf[0])

            # 🔴 CONFIDENCE FILTER (Respecting cfg)
            if conf < self.cfg.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w-1, x2); y2 = min(h-1, y2)

            width = x2 - x1
            height = y2 - y1

            area = width * height
            area_ratio = area / max(self._frame_area, 1)

            # 🔴 AREA FILTER (Exactly matches exam_proctoring_system.py)
            if area_ratio < self.cfg.min_phone_area_ratio or area_ratio > self.cfg.max_phone_area_ratio:
                continue

            # 🔴 SPATIAL DISTANCE FILTER
            dist = float("inf")
            if face_centroid_px:
                fx, fy = face_centroid_px
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                dist = float(np.sqrt((cx - fx)**2 + (cy - fy)**2))

                if dist > diag * self.cfg.phone_face_proximity_factor:
                    continue  # Phone is background object — skip

            # 🔴 CLASSIFIER FILTER
            if getattr(self.cfg, "use_classifier", False):
                if not self._classify_crop(frame, (x1, y1, x2, y2)):
                    continue

            result.detections.append(
                PhoneDetection((x1, y1, x2, y2), conf, area_ratio, dist)
            )

        result.filtered_count = len(result.detections)
        return result

    # ─────────────────────────────────────────────────────────

    def draw_detections(self, frame, result, color):

        for det in result.detections:
            x1, y1, x2, y2 = det.bbox_px

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"PHONE {det.confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
