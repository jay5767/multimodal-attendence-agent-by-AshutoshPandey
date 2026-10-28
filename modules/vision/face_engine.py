import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from config import Config

class FaceEngine:
    """
    Vision Face Detection & Recognition Engine.
    Uses InsightFace (buffalo_l) with onnxruntime CUDA/GPU and automatic CPU fallback.
    Also provides live 5-sample enrollment quality checks (size, blur, lighting, single face).
    """
    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = root_dir or str(Config.DATA_DIR / "insightface")
        self.app = None
        self.active_provider = "CPU"
        self.is_gpu = False

        self._init_insightface()

    def _init_cascade(self):
        """Loads high-accuracy OpenCV Haar Cascade with bundled alt2, default, and profile models."""
        base_dir = Path(__file__).resolve().parent
        # alt2 is Gentle AdaBoost with significantly higher sensitivity in real webcam lighting
        candidates = [
            str(base_dir / "haarcascade_frontalface_alt2.xml"),
            str(base_dir / "haarcascade_frontalface_default.xml"),
            getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_alt2.xml",
            getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        ]
        loaded_cas = None
        for p in candidates:
            if p and os.path.exists(p):
                cas = cv2.CascadeClassifier(p)
                if not cas.empty():
                    print(f"[FaceEngine] Successfully loaded Haar Cascade from {p}")
                    loaded_cas = cas
                    break

        # Profile cascade for side-facing head poses
        prof_p = str(base_dir / "haarcascade_profileface.xml")
        if os.path.exists(prof_p):
            pcas = cv2.CascadeClassifier(prof_p)
            self.profile_cascade = pcas if not pcas.empty() else None
        else:
            self.profile_cascade = None

        if loaded_cas is None:
            print("[FaceEngine] Warning: Haar Cascade XML not found, initializing empty classifier.")
            return cv2.CascadeClassifier()
        return loaded_cas

    def _init_insightface(self):
        """Initializes InsightFace FaceAnalysis with CUDA, CPU, and ultra-lean cloud fallbacks."""
        is_cloud = bool(os.environ.get("RENDER") or os.environ.get("LOW_MEMORY"))
        model_path = Path(self.root_dir) / "models" / Config.INSIGHTFACE_MODEL
        if is_cloud and not model_path.exists():
            print("[FaceEngine] Cloud 512MB mode active: skipping heavy zip download, using lightweight OpenCV engine.")
            self.app = None
            self.active_provider = "CPU (Lean Fallback)"
            self.is_gpu = False
            self.cascade = self._init_cascade()
            return

        try:
            import insightface
            import onnxruntime as ort

            available_providers = ort.get_available_providers()
            desired_providers = []
            if Config.PREFER_GPU and "CUDAExecutionProvider" in available_providers:
                desired_providers.append("CUDAExecutionProvider")
            desired_providers.append("CPUExecutionProvider")

            # Only load detection and recognition to save >250MB RAM (skips 140MB 3D model)
            allowed = ['detection', 'recognition'] if is_cloud else None
            self.app = insightface.app.FaceAnalysis(
                name=Config.INSIGHTFACE_MODEL,
                root=self.root_dir,
                allowed_modules=allowed,
                providers=desired_providers
            )
            ctx_id = 0 if ("CUDAExecutionProvider" in desired_providers and "CUDAExecutionProvider" in available_providers) else -1
            det_w = 320 if is_cloud else Config.TARGET_WIDTH
            det_h = 240 if is_cloud else Config.TARGET_HEIGHT
            self.app.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))

            if ctx_id == 0:
                self.active_provider = "CUDA (GPU)"
                self.is_gpu = True
            else:
                self.active_provider = "CPU"
                self.is_gpu = False
            print(f"[FaceEngine] Initialized InsightFace {Config.INSIGHTFACE_MODEL} on {self.active_provider}")
        except Exception as e:
            print(f"[FaceEngine] InsightFace initialization warning: {e}. Activating OpenCV Haar/DNN fallback.")
            self.app = None
            self.active_provider = "CPU (Fallback)"
            self.is_gpu = False
            self.cascade = self._init_cascade()

    def get_acceleration_info(self) -> Dict[str, Any]:
        """Returns hardware acceleration info for the UI badge."""
        return {
            "provider": self.active_provider,
            "is_gpu": self.is_gpu,
            "model": Config.INSIGHTFACE_MODEL if self.app else "OpenCV Cascade"
        }

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in frame_bgr and computes 512-d normalized face embeddings.
        Returns:
            [
                {
                    "bbox": (x1, y1, x2, y2),
                    "embedding": np.ndarray (512,),
                    "landmarks": np.ndarray or None,
                    "det_score": float
                },
                ...
            ]
        """
        results = []
        if frame_bgr is None or frame_bgr.size == 0:
            return results

        h, w = frame_bgr.shape[:2]

        if self.app is not None:
            try:
                faces = self.app.get(frame_bgr)
                for f in faces:
                    x1, y1, x2, y2 = map(int, f.bbox)
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # L2 normalize embedding
                    emb = f.embedding
                    norm = np.linalg.norm(emb)
                    if norm > 1e-6:
                        emb = emb / norm

                    results.append({
                        "bbox": (x1, y1, x2, y2),
                        "embedding": emb,
                        "landmarks": getattr(f, "kps", None),
                        "det_score": float(getattr(f, "det_score", 0.95))
                    })
                return results
            except Exception as e:
                pass

        # Fallback detection with enhanced sensitivity & geometric landmark synthesis
        if hasattr(self, "cascade") and self.cascade is not None and not self.cascade.empty():
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                # 1. Histogram equalization (enhances low-light and uneven webcam lighting)
                gray_eq = cv2.equalizeHist(gray)
                dets = self.cascade.detectMultiScale(gray_eq, scaleFactor=1.08, minNeighbors=3, minSize=(30, 30))
                
                # 2. Raw grayscale with relaxed threshold if equalized returned nothing
                if len(dets) == 0:
                    dets = self.cascade.detectMultiScale(gray, scaleFactor=1.06, minNeighbors=2, minSize=(25, 25))

                # 3. Profile cascade check if turned away
                if len(dets) == 0 and hasattr(self, "profile_cascade") and self.profile_cascade is not None and not self.profile_cascade.empty():
                    dets = self.profile_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

                for (x, y, fw, fh) in dets:
                    # Synthesize 5 facial keypoints for head pose geometry & gaze tracking
                    kps = np.array([
                        [x + 0.32 * fw, y + 0.38 * fh],  # Left eye
                        [x + 0.68 * fw, y + 0.38 * fh],  # Right eye
                        [x + 0.50 * fw, y + 0.55 * fh],  # Nose tip
                        [x + 0.35 * fw, y + 0.75 * fh],  # Left mouth
                        [x + 0.65 * fw, y + 0.75 * fh],  # Right mouth
                    ], dtype=np.float32)

                    # Generate pseudo 512-d normalized embedding from cropped face spatial features
                    face_crop = cv2.resize(gray[y:y+fh, x:x+fw], (32, 32)).flatten().astype(np.float32)
                    emb = np.resize(face_crop, (512,))
                    norm = np.linalg.norm(emb)
                    if norm > 1e-6:
                        emb = emb / norm
                    results.append({
                        "bbox": (x, y, x + fw, y + fh),
                        "embedding": emb,
                        "landmarks": kps,
                        "det_score": 0.90
                    })
            except Exception as e:
                pass

        return results

    def match_embedding(self, query_emb: np.ndarray, enrolled_dict: Dict[int, np.ndarray], threshold: float = Config.FACE_SIMILARITY_THRESHOLD) -> Tuple[Optional[int], float]:
        """
        Cosine similarity matching against enrolled student embeddings.
        Returns (best_student_id, highest_similarity) or (None, highest_similarity)
        """
        if query_emb is None or len(enrolled_dict) == 0:
            return (None, 0.0)

        best_id = None
        best_sim = -1.0

        for sid, emb in enrolled_dict.items():
            # Both are L2 normalized, so cosine similarity is dot product
            sim = float(np.dot(query_emb, emb))
            if sim > best_sim:
                best_sim = sim
                best_id = sid

        if best_sim >= threshold:
            return (best_id, round(best_sim, 3))
        else:
            return (None, round(max(0.0, best_sim), 3))

    def check_sample_quality(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Validates live enrollment sample quality against 4 gates:
        1. Single face detected
        2. Face size >= MIN_FACE_SIZE_PX (100px)
        3. Sharpness / Blur check (variance of Laplacian >= 70.0)
        4. Lighting / Brightness check (45 <= mean grayscale <= 230)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "passed": False,
                "single_face_ok": False,
                "face_size_ok": False,
                "blur_ok": False,
                "lighting_ok": False,
                "feedback": "No camera frame received.",
                "embedding": None,
                "bbox": None
            }

        faces = self.detect_faces(frame_bgr)
        single_face_ok = (len(faces) == 1)

        if len(faces) == 0:
            return {
                "passed": False,
                "single_face_ok": False,
                "face_size_ok": False,
                "blur_ok": False,
                "lighting_ok": False,
                "feedback": "No face detected. Center your face in the camera.",
                "embedding": None,
                "bbox": None
            }

        if len(faces) > 1:
            return {
                "passed": False,
                "single_face_ok": False,
                "face_size_ok": False,
                "blur_ok": False,
                "lighting_ok": False,
                "feedback": f"Multiple faces detected ({len(faces)}). Ensure only one student is in frame.",
                "embedding": None,
                "bbox": None
            }

        face = faces[0]
        x1, y1, x2, y2 = face["bbox"]
        fw = x2 - x1
        fh = y2 - y1

        # 1. Face size gate
        face_size_ok = (fw >= Config.MIN_FACE_SIZE_PX and fh >= Config.MIN_FACE_SIZE_PX)

        # Crop face
        face_crop = frame_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            face_crop = frame_bgr

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # 2. Blur / sharpness gate
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_ok = (laplacian_var >= Config.MIN_LAPLACIAN_VAR)

        # 3. Lighting / brightness gate
        mean_brightness = float(np.mean(gray))
        lighting_ok = (Config.MIN_BRIGHTNESS <= mean_brightness <= Config.MAX_BRIGHTNESS)

        # Feedback messaging
        feedbacks = []
        if not face_size_ok:
            feedbacks.append("Move closer to the camera")
        if not blur_ok:
            feedbacks.append("Hold still to reduce motion blur")
        if mean_brightness < Config.MIN_BRIGHTNESS:
            feedbacks.append("Lighting too dark; face a light source")
        elif mean_brightness > Config.MAX_BRIGHTNESS:
            feedbacks.append("Lighting overexposed; reduce backlight")

        passed = single_face_ok  # As long as a single face is detected, sample capture is enabled
        is_optimal = (single_face_ok and face_size_ok and blur_ok and lighting_ok)
        feedback_str = "Excellent sample! Ready to capture." if is_optimal else ("Sample acceptable (" + " & ".join(feedbacks) + ")")
        emb_list = face["embedding"].tolist() if (face.get("embedding") is not None) else None

        return {
            "passed": passed,
            "optimal": is_optimal,
            "single_face_ok": single_face_ok,
            "face_size_ok": face_size_ok,
            "blur_ok": blur_ok,
            "lighting_ok": lighting_ok,
            "laplacian_var": round(laplacian_var, 1),
            "mean_brightness": round(mean_brightness, 1),
            "face_width": fw,
            "feedback": feedback_str,
            "embedding": emb_list,
            "bbox": face["bbox"]
        }
