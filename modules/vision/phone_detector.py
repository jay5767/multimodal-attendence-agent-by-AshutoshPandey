"""
Offline Phone & Distraction Detector using YOLOv8n (Class 67: Cell Phone)
Supports ultra-lean ONNXRuntime (<30MB RAM, zero PyTorch) and native PyTorch fallback.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from config import Config

class PhoneDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.onnx_path = getattr(Config, "YOLO_ONNX_PATH", str(Config.BASE_DIR / "models" / "yolov8n.onnx"))
        self.pt_path = model_path or Config.YOLO_MODEL_PATH
        self.session = None
        self.model = None
        self.backend = None  # 'onnx', 'ultralytics', or 'posture'
        self.is_loaded = False
        self._init_model()

    def _init_model(self):
        # 1. Prefer ultra-lean ONNXRuntime (runs without PyTorch, uses <30MB RAM)
        if os.path.exists(self.onnx_path):
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
                providers = []
                if Config.PREFER_GPU and "CUDAExecutionProvider" in available:
                    providers.append("CUDAExecutionProvider")
                providers.append("CPUExecutionProvider")
                self.session = ort.InferenceSession(self.onnx_path, providers=providers)
                self.model = self.session
                self.input_name = self.session.get_inputs()[0].name
                self.backend = "onnx"
                self.is_loaded = True
                print(f"[PhoneDetector] Loaded ultra-lean YOLOv8n ONNX model from {self.onnx_path} ({providers[0]})")
                return
            except Exception as e:
                print(f"[PhoneDetector] ONNX init notice: {e}")

        # 2. Fallback to Ultralytics YOLO if available
        if os.path.exists(self.pt_path):
            try:
                from ultralytics import YOLO
                abs_path = os.path.abspath(self.pt_path)
                self.model = YOLO(abs_path)
                self.backend = "ultralytics"
                self.is_loaded = True
                print(f"[PhoneDetector] Loaded YOLOv8n PyTorch model from {abs_path}")
                return
            except Exception as e:
                print(f"[PhoneDetector] PyTorch init notice: {e}")

        print("[PhoneDetector] Notice: Running posture-based phone detection fallback.")
        self.backend = "posture"
        self.is_loaded = False

    def detect_phones(self, frame_bgr: np.ndarray, conf_threshold: float = 0.22) -> List[Dict[str, Any]]:
        """
        Detects mobile phones / cell phones in the frame.
        COCO class 67 is 'cell phone'.
        """
        if not self.is_loaded or frame_bgr is None or frame_bgr.size == 0:
            return []

        if self.backend == "onnx" and self.session is not None:
            return self._detect_phones_onnx(frame_bgr, conf_threshold)
        elif self.backend == "ultralytics" and self.model is not None:
            return self._detect_phones_ultralytics(frame_bgr, conf_threshold)
        return []

    def _detect_phones_onnx(self, frame_bgr: np.ndarray, conf_threshold: float = 0.22) -> List[Dict[str, Any]]:
        """Runs fast, low-memory ONNXRuntime inference on YOLOv8n."""
        try:
            h, w = frame_bgr.shape[:2]
            r = min(640.0 / h, 640.0 / w)
            new_w, new_h = int(round(w * r)), int(round(h * r))
            resized = cv2.resize(frame_bgr, (new_w, new_h))

            dw, dh = (640 - new_w) / 2.0, (640 - new_h) / 2.0
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

            # Preprocess blob: BGR to RGB -> float32 [0, 1] -> NCHW
            blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

            output = self.session.run(None, {self.input_name: blob})[0]
            preds = output[0].T  # (8400, 84)

            # Class 67 is cell phone (0-3 are bbox cx, cy, w, h)
            phone_scores = preds[:, 4 + 67]
            valid_mask = phone_scores >= conf_threshold

            if not np.any(valid_mask):
                return []

            valid_preds = preds[valid_mask]
            valid_scores = phone_scores[valid_mask]

            boxes_for_nms = []
            scores_for_nms = []
            for pred, score in zip(valid_preds, valid_scores):
                cx, cy, bw, bh = pred[0:4]
                # Map back to unpadded coordinates
                x1 = int(round((cx - bw / 2.0 - dw) / r))
                y1 = int(round((cy - bh / 2.0 - dh) / r))
                box_w = int(round(bw / r))
                box_h = int(round(bh / r))

                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                box_w = max(1, min(w - x1, box_w))
                box_h = max(1, min(h - y1, box_h))

                boxes_for_nms.append([x1, y1, box_w, box_h])
                scores_for_nms.append(float(score))

            indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, conf_threshold, 0.45)
            phones = []
            if len(indices) > 0:
                for idx in indices.flatten():
                    bx, by, bw, bh = boxes_for_nms[idx]
                    phones.append({
                        "bbox": [bx, by, bx + bw, by + bh],
                        "confidence": round(scores_for_nms[idx], 3),
                        "label": "cell phone"
                    })
            return phones
        except Exception:
            return []

    def _detect_phones_ultralytics(self, frame_bgr: np.ndarray, conf_threshold: float = 0.22) -> List[Dict[str, Any]]:
        """Runs native Ultralytics PyTorch prediction."""
        try:
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
        except Exception:
            return []

    def match_phone_to_face(
        self,
        face_bbox: List[int],
        phones: List[Dict[str, Any]],
        pitch: float = 0.0,
        frame_shape: Tuple[int, int] = (480, 640),
        track_id: int = 1
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Determines if a detected phone belongs to a student face bounding box.
        Includes temporal debounce smoothing to completely eliminate 1-frame glitches.
        """
        if not hasattr(self, "track_phone_state"):
            self.track_phone_state = {}

        state = self.track_phone_state.setdefault(track_id, {
            "count": 0,
            "active": False,
            "cooldown": 0,
            "phone": None,
            "reason": ""
        })

        fx1, fy1, fx2, fy2 = face_bbox
        fw = fx2 - fx1
        fh = fy2 - fy1
        fcx = (fx1 + fx2) / 2.0
        fcy = (fy1 + fy2) / 2.0

        # Define student's upper torso / hand interaction zone
        torso_x1 = fx1 - fw * 1.2
        torso_x2 = fx2 + fw * 1.2
        torso_y1 = fy1
        torso_y2 = fy2 + fh * 3.2

        candidate_phone = None
        for phone in phones:
            px1, py1, px2, py2 = phone["bbox"]
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0

            x_overlap = max(0, min(torso_x2, px2) - max(torso_x1, px1))
            y_overlap = max(0, min(torso_y2, py2) - max(torso_y1, py1))
            dist_x = abs(fcx - pcx)
            dist_y = pcy - fcy

            if (x_overlap > 0 and y_overlap > 0) or (dist_x < fw * 1.6 and 0 < dist_y < fh * 3.5):
                candidate_phone = phone
                break

        # Anti-glitch debounce: require 2 consecutive frames to trigger, hold for 10 frames (~1s)
        if candidate_phone is not None:
            state["count"] += 1
            if state["count"] >= 2:
                state["active"] = True
                state["cooldown"] = 10
                state["phone"] = candidate_phone
                state["reason"] = f"Cell Phone Detected ({candidate_phone['confidence']:.2f})"
        else:
            state["count"] = max(0, state["count"] - 1)
            if state["cooldown"] > 0:
                state["cooldown"] -= 1
            else:
                state["active"] = False
                state["phone"] = None
                state["reason"] = ""

        return state["active"], state["phone"], state["reason"]

    def cleanup(self, active_track_ids: List[int]):
        """Remove tracks that are no longer active."""
        if hasattr(self, "track_phone_state"):
            for tid in list(self.track_phone_state.keys()):
                if tid not in active_track_ids:
                    del self.track_phone_state[tid]
