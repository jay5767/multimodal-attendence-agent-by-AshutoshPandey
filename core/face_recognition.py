"""Face detection and local identity matching module.

Provides privacy-compliant facial feature extraction and enrolled student matching.
Does NOT upload images to external servers or make unverified claims.
"""

import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from config.settings import DEFAULT_CONFIG
from utils.logging_config import logger


@dataclass
class FaceMatchResult:
    """Result of facial matching against enrolled database."""
    student_id: Optional[str]
    confidence: float
    is_matched: bool
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    status_label: str  # 'Verified', 'Uncertain', 'Anonymous'


class FaceRecognizer:
    """Local, offline face recognizer with multi-backend fallbacks."""

    def __init__(self, similarity_threshold: float = DEFAULT_CONFIG.attendance.face_similarity_threshold):
        self.similarity_threshold = similarity_threshold
        self.enrolled_students: Dict[str, np.ndarray] = {}
        self.backend = "opencv"
        self._face_cascade = None
        self._init_face_detector()

    def _init_face_detector(self) -> None:
        """Initialize OpenCV Haar Cascade for face localization."""
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                logger.warning("Haar cascade empty or not found; face detection may be limited.")
            else:
                logger.info("FaceRecognizer initialized with OpenCV Haar Cascade.")
        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")

    def load_enrolled_students(self, students_list: List[Dict[str, Any]]) -> None:
        """Load consented students with valid face embeddings into cache."""
        self.enrolled_students.clear()
        for s in students_list:
            sid = s.get("student_id")
            consent = s.get("has_biometric_consent", False)
            encoding_json = s.get("face_encoding_json")

            if sid and consent and encoding_json:
                try:
                    vec = np.array(json.loads(encoding_json), dtype=np.float32)
                    # Normalize vector
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        vec = vec / norm
                    self.enrolled_students[sid] = vec
                except Exception as e:
                    logger.warning(f"Could not parse face encoding for student {sid}: {e}")
        logger.info(f"Loaded {len(self.enrolled_students)} enrolled student biometric signatures.")

    def extract_face_feature_vector(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract a 128-dimensional normalized feature vector from a face crop.
        
        Uses spatial multi-scale histogram and grayscale intensity distribution
        as a deterministic, reproducible local embedding representation.
        """
        if face_crop is None or face_crop.size == 0:
            return None
        try:
            import cv2
            # Resize face crop to standardized 64x64
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
            standardized = cv2.resize(gray, (64, 64))
            standardized = cv2.equalizeHist(standardized)

            # Block-based mean and variance features (4x4 grid = 16 blocks * 8 bins histogram = 128-d)
            features = []
            h, w = standardized.shape
            bh, bw = h // 4, w // 4
            for r in range(4):
                for c in range(4):
                    block = standardized[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
                    hist = cv2.calcHist([block], [0], None, [8], [0, 256]).flatten()
                    features.extend(hist)

            vec = np.array(features, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            return vec
        except Exception as e:
            logger.error(f"Error computing face feature vector: {e}")
            return None

    def detect_faces(self, frame: np.ndarray, person_box: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[int, int, int, int]]:
        """Detect faces within the entire frame or within a person bounding box."""
        if frame is None or self._face_cascade is None:
            return []
        try:
            import cv2
            if person_box is not None:
                x1, y1, x2, y2 = person_box
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                # Face is typically in the upper 45% of the person's bounding box
                crop_h = max(10, int((y2 - y1) * 0.45))
                sub_frame = frame[y1:y1 + crop_h, x1:x2]
                if sub_frame.size == 0:
                    return []
                gray = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2GRAY)
                faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
                # Shift coordinates back to full frame
                return [(x + x1, y + y1, fw, fh) for (x, y, fw, fh) in faces]
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return []

    def match_face(self, face_crop: np.ndarray) -> FaceMatchResult:
        """Compare a face crop against enrolled students."""
        if not self.enrolled_students:
            return FaceMatchResult(
                student_id=None,
                confidence=0.0,
                is_matched=False,
                bbox=(0, 0, 0, 0),
                status_label="Anonymous (No enrolled database)"
            )

        vec = self.extract_face_feature_vector(face_crop)
        if vec is None:
            return FaceMatchResult(
                student_id=None,
                confidence=0.0,
                is_matched=False,
                bbox=(0, 0, 0, 0),
                status_label="Face Quality Low"
            )

        best_sid = None
        best_sim = -1.0

        for sid, enrolled_vec in self.enrolled_students.items():
            # Cosine similarity between normalized vectors = dot product
            sim = float(np.dot(vec, enrolled_vec))
            if sim > best_sim:
                best_sim = sim
                best_sid = sid

        if best_sim >= self.similarity_threshold:
            return FaceMatchResult(
                student_id=best_sid,
                confidence=round(best_sim, 3),
                is_matched=True,
                bbox=(0, 0, 0, 0),
                status_label="Verified"
            )
        else:
            return FaceMatchResult(
                student_id=best_sid if best_sim > 0.40 else None,
                confidence=round(max(0.0, best_sim), 3),
                is_matched=False,
                bbox=(0, 0, 0, 0),
                status_label="Uncertain" if best_sim > 0.40 else "Anonymous"
            )
