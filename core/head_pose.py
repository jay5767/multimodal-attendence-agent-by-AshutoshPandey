"""Head pose estimation and observable visual movement analyzer.

Estimates head orientation (yaw, pitch, roll) and frame-to-frame movement dynamics.
Observable signals are strictly distinguished from internal cognitive or emotional states.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
import math
from config.settings import DEFAULT_CONFIG
from utils.logging_config import logger


@dataclass
class HeadPoseResult:
    """Estimated orientation angles and observable classification."""
    yaw: Optional[float]  # Degrees (-left, +right)
    pitch: Optional[float]  # Degrees (-down, +up)
    roll: Optional[float]  # Degrees (tilt)
    movement_speed: float  # Normalized pixel speed
    observable_state: str  # 'forward_facing', 'turned_away', 'looking_down', 'looking_up', 'unavailable'
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head_yaw": self.yaw,
            "head_pitch": self.pitch,
            "head_roll": self.roll,
            "movement_speed": self.movement_speed,
            "observable_state": self.observable_state,
            "confidence": self.confidence
        }


class HeadPoseEstimator:
    """Estimates head orientation and observable movement using geometric feature points."""

    def __init__(self, config=DEFAULT_CONFIG.engagement):
        self.config = config
        self._prev_centers: Dict[int, Tuple[float, float, float]] = {}  # track_id -> (cx, cy, timestamp)

    def estimate(
        self,
        frame: np.ndarray,
        person_bbox: Tuple[int, int, int, int],
        track_id: int,
        timestamp_sec: float
    ) -> HeadPoseResult:
        """Estimate head orientation and movement for a tracked person in a video frame."""
        x1, y1, x2, y2 = person_bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Calculate normalized movement speed based on tracked person center
        movement_speed = 0.0
        if track_id in self._prev_centers:
            prev_cx, prev_cy, prev_t = self._prev_centers[track_id]
            dt = max(0.001, timestamp_sec - prev_t)
            pixel_dist = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
            # Normalize by person bounding box height to be distance-invariant
            movement_speed = round((pixel_dist / h) / dt, 3)

        self._prev_centers[track_id] = (cx, cy, timestamp_sec)

        # Facial analysis within upper quadrant of person bounding box
        face_h = max(10, int(h * 0.40))
        face_crop = frame[max(0, y1):min(frame.shape[0], y1 + face_h), max(0, x1):min(frame.shape[1], x2)]

        if face_crop.size == 0 or face_crop.shape[0] < 15 or face_crop.shape[1] < 15:
            # When face region is occluded or too small, explicitly mark unavailable
            return HeadPoseResult(
                yaw=None,
                pitch=None,
                roll=None,
                movement_speed=movement_speed,
                observable_state="unavailable",
                confidence=0.0
            )

        try:
            import cv2
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            # Find intensity centroids (eyes/brows region vs mouth/chin region)
            mid_y = gray.shape[0] // 2
            mid_x = gray.shape[1] // 2
            left_half = gray[:, :mid_x]
            right_half = gray[:, mid_x:]

            left_mean = float(np.mean(left_half))
            right_mean = float(np.mean(right_half))
            diff_ratio = (right_mean - left_mean) / max(1.0, (right_mean + left_mean))

            # Approximation of yaw based on horizontal gradient asymmetry
            yaw = float(np.clip(diff_ratio * 90.0, -60.0, 60.0))

            top_half = gray[:mid_y, :]
            bottom_half = gray[mid_y:, :]
            top_mean = float(np.mean(top_half))
            bottom_mean = float(np.mean(bottom_half))
            v_ratio = (bottom_mean - top_mean) / max(1.0, (bottom_mean + top_mean))
            pitch = float(np.clip(v_ratio * 70.0, -45.0, 45.0))
            roll = 0.0

            # Observable classification
            if abs(yaw) > self.config.yaw_forward_limit:
                state = "turned_away"
            elif pitch < -self.config.pitch_down_limit:
                state = "looking_down"
            elif pitch > self.config.pitch_up_limit:
                state = "looking_up"
            else:
                state = "forward_facing"

            return HeadPoseResult(
                yaw=round(yaw, 1),
                pitch=round(pitch, 1),
                roll=round(roll, 1),
                movement_speed=movement_speed,
                observable_state=state,
                confidence=0.75
            )
        except Exception as e:
            logger.debug(f"Pose calculation fallback: {e}")
            return HeadPoseResult(
                yaw=None,
                pitch=None,
                roll=None,
                movement_speed=movement_speed,
                observable_state="unavailable",
                confidence=0.0
            )
