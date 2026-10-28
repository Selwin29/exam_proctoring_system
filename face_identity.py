import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class FaceIdentityVerifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reference_embedding: Optional[np.ndarray] = None
        self.reference_samples = []
        self.registration_complete = False
        
        # We will lazy-load deepface in the thread to avoid main-thread blocking
        self._DeepFace = None

    def set_reference_embedding(self, embedding: np.ndarray):
        """Bypass the 30-frame local collection phase if backend provides embedding."""
        self.reference_embedding = embedding
        self.registration_complete = True
        print("\n[FaceIdentity] Reference Face injected from Backend.\n")

    def _ensure_model_loaded(self):
        if self._DeepFace is None and self.cfg.enabled:
            print("[Init] Loading DeepFace model for proxy detection...")
            try:
                from deepface import DeepFace
                self._DeepFace = DeepFace
                # Dummy call to initialize the model into memory
                dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
                self._DeepFace.represent(img_path=dummy_img, model_name=self.cfg.model_name, enforce_detection=False)
                print("[Init] DeepFace proxy detection ready.")
            except ImportError:
                print("[Error] DeepFace is not installed. Run `pip install deepface`. Proxy detection disabled.")
                self.cfg.enabled = False
            except Exception as e:
                print(f"[Error] Failed to initialize DeepFace: {e}")
                self.cfg.enabled = False

    def _extract_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        self._ensure_model_loaded()
        if not self.cfg.enabled:
            return None
            
        try:
            # We enforce_detection=False because MediaPipe already guarantees a face crop
            reps = self._DeepFace.represent(img_path=face_crop, model_name=self.cfg.model_name, enforce_detection=False)
            if len(reps) > 0:
                return np.array(reps[0]['embedding'])
        except Exception as e:
            logger.warning(f"Embedding extraction failed: {e}")
        return None

    def _compute_cosine_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        # Distance calculation
        a = np.matmul(np.transpose(emb1), emb2)
        b = np.sum(np.multiply(emb1, emb1))
        c = np.sum(np.multiply(emb2, emb2))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def process(self, frame: np.ndarray, bbox_norm: Optional[tuple]) -> bool:
        """
        Processes a frame and crop to determine if it's a proxy.
        Returns `True` if it is a DIFFERENT person (Proxy Detected).
        Returns `False` if it's the registered user, or if registration isn't done.
        """
        if not self.cfg.enabled or bbox_norm is None:
            return False

        h, w = frame.shape[:2]
        x1n, y1n, x2n, y2n = bbox_norm
        
        # Add a little margin
        x1 = max(0, int(x1n * w) - 20)
        y1 = max(0, int(y1n * h) - 20)
        x2 = min(w, int(x2n * w) + 20)
        y2 = min(h, int(y2n * h) + 20)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0 or face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
            return False # Invalid crop

        embedding = self._extract_embedding(face_crop)
        if embedding is None:
            return False

        # Phase 1: Registration
        if not self.registration_complete:
            self.reference_samples.append(embedding)
            if len(self.reference_samples) >= self.cfg.reference_collection_frames:
                self.reference_embedding = np.mean(self.reference_samples, axis=0)
                self.registration_complete = True
                print("\n[FaceIdentity] Reference Face Registered successfully.\n")
            return False
            
        # Phase 2: Verification
        distance = self._compute_cosine_distance(self.reference_embedding, embedding)
        
        # If distance > threshold, they are too different -> Proxy
        if distance > self.cfg.similarity_threshold:
            return True
            
        return False


def extract_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Module level helper specifically designed for app.py to extract 
    a highly accurate reference embedding using enforcing mode.
    """
    from deepface import DeepFace
    import cv2
    try:
        import logging
        logging.getLogger(__name__).info("Extracting reference face from web capture...")
        
        # UI sends full 640x480 frame. The face is centered in the UI oval.
        # Crop the central region (about 60% of width/height).
        img = cv2.imread(image_path)
        if img is None: return None
        h, w = img.shape[:2]
        
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.15) # Face tends to be slightly higher, less top margin
        bottom_margin_y = int(h * 0.25)
        
        crop = img[margin_y:h-bottom_margin_y, margin_x:w-margin_x]
        
        # Feed the cropped face array directly with enforce_detection=False. 
        # DeepFace default opencv cascade is unreliable, so bridging directly solves fails.
        reps = DeepFace.represent(img_path=crop, model_name="Facenet", enforce_detection=False)
        if len(reps) > 0:
            return np.array(reps[0]['embedding'])
    except Exception as e:
        print(f"Face extraction failed: {e}")
    return None
