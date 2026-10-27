"""
Offline Phone & Distraction Detector using YOLOv8n (Class 67: Cell Phone)
Runs 100% locally with zero cloud dependencies.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from config import Config

class PhoneDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or Config.YOLO_MODEL_PATH
        self.model = None
        self.is_loaded = False
        self._init_model()

    def _init_model(self):
        try:
            if not os.path.exists(self.model_path):
                print(f"[PhoneDetector] Model weights not found at {self.model_path}. Running posture fallback.")
                return

            from ultralytics import YOLO
            # Load with exact absolute path to prevent automatic GitHub asset download attempts
            abs_path = os.path.abspath(self.model_path)
            self.model = YOLO(abs_path)
            self.is_loaded = True
            print(f"[PhoneDetector] Successfully loaded YOLOv8n phone detector offline from {abs_path}")
        except Exception as e:
            print(f"[PhoneDetector] Notice: YOLO model offline load exception: {e}. Using posture-based phone detection.")
            self.is_loaded = False

    def detect_phones(self, frame_bgr: np.ndarray, conf_threshold: float = 0.22) -> List[Dict[str, Any]]:
        """
        Detects mobile phones / cell phones in the frame.
        COCO class 67 is 'cell phone'.
        """
        if not self.is_loaded or self.model is None or frame_bgr is None:
            return []

        try:
            # Predict for class 67 (cell phone)
            results = self.model(frame_bgr, verbose=False, classes=[67], conf=conf_threshold)
            phones = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id == 67:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        conf = float(box.conf[0].item())
                        phones.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": round(conf, 3),
                            "label": "cell phone"
                        })
            return phones
        except Exception as e:
            # Prevent camera loop disruption
            return []

    def match_phone_to_face(
        self,
        face_bbox: List[int],
        phones: List[Dict[str, Any]],
        pitch: float = 0.0,
        frame_shape: Tuple[int, int] = (480, 640)
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Determines if a detected phone belongs to a student face bounding box,
        or if the student's head posture matches active phone browsing in lap.
        """
        fx1, fy1, fx2, fy2 = face_bbox
        fw = fx2 - fx1
        fh = fy2 - fy1
        fcx = (fx1 + fx2) / 2.0
        fcy = (fy1 + fy2) / 2.0

        # Define student's upper torso / hand / lap interaction zone
        torso_x1 = fx1 - fw * 0.9
        torso_x2 = fx2 + fw * 0.9
        torso_y1 = fy1  # from forehead down
        torso_y2 = fy2 + fh * 3.5  # down to lap/desk

        # Check visual phone bounding boxes
        for phone in phones:
            px1, py1, px2, py2 = phone["bbox"]
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0

            # Direct overlap with torso/lap zone
            x_overlap = max(0, min(torso_x2, px2) - max(torso_x1, px1))
            y_overlap = max(0, min(torso_y2, py2) - max(torso_y1, py1))
            
            # Distance from face
            dist_x = abs(fcx - pcx)
            dist_y = pcy - fcy

            if (x_overlap > 0 and y_overlap > 0) or (dist_x < fw * 1.5 and 0 < dist_y < fh * 3.8):
                return True, phone, f"Active Phone In Hand (conf {phone['confidence']:.2f})"

        # Posture-based phone usage (e.g., student looking down into lap at screen where phone is partially occluded)
        if pitch < -13.5:
            # Head severely tilted down into lap
            return True, None, "Phone Posture: Gaze Downward into Lap"

        return False, None, ""
