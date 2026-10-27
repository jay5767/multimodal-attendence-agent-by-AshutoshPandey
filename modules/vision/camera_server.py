import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

class ServerCameraStream:
    """
    Direct server-side OpenCV camera capture with multi-index hardware detection
    (prioritizing index 2 with CAP_DSHOW for integrated laptop webcams).
    """
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.lock = threading.Lock()
        self.last_frame: Optional[np.ndarray] = None
        self.device_index = 2
        self.api_preference = cv2.CAP_DSHOW
        self.fps = 0.0

    def find_working_device(self) -> Tuple[int, int]:
        """Probes indices [2, 0, 1] to find working physical camera."""
        for idx in [2, 0, 1]:
            for api in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                try:
                    c = cv2.VideoCapture(idx, api)
                    if c.isOpened():
                        ret, f = c.read()
                        if ret and f is not None and f.size > 0:
                            c.release()
                            return (idx, api)
                        c.release()
                except Exception:
                    pass
        return (2, cv2.CAP_DSHOW)

    def start(self, device_index: Optional[int] = None) -> bool:
        """Starts background hardware capture thread."""
        with self.lock:
            if self.is_running and self.cap and self.cap.isOpened():
                return True

            target_idx = device_index
            target_api = self.api_preference
            if target_idx is None:
                target_idx, target_api = self.find_working_device()

            try:
                self.cap = cv2.VideoCapture(target_idx, target_api)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(target_idx) # Fallback to default API

                if not self.cap.isOpened():
                    return False

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.device_index = target_idx
                self.is_running = True

                self.thread = threading.Thread(target=self._capture_worker, daemon=True)
                self.thread.start()
                return True
            except Exception as e:
                print(f"[ServerCamera] Error opening camera {target_idx}: {e}")
                self.is_running = False
                return False

    def _capture_worker(self):
        frame_count = 0
        t_start = time.time()
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.last_frame = frame
                frame_count += 1
                dt = time.time() - t_start
                if dt >= 1.0:
                    self.fps = round(frame_count / dt, 1)
                    frame_count = 0
                    t_start = time.time()
            time.sleep(0.02) # ~40 FPS capture rate

    def read_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.last_frame is not None:
                return self.last_frame.copy()
            return None

    def stop(self):
        with self.lock:
            self.is_running = False
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
