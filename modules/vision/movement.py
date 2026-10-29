import time
from typing import Dict, List, Tuple
import numpy as np

class MovementAnalyzer:
    """
    Computes landmark displacement, optical movement score, and inactivity duration
    per tracked face over time intervals.
    """
    def __init__(self, movement_sensitivity: float = 1.0):
        self.sensitivity = movement_sensitivity
        self.track_data: Dict[int, Dict] = {}

    def update_track(self, track_id: int, current_landmarks: List[Tuple[float, float]], bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Updates track displacement and returns:
            (movement_score: 0.0 to 1.0, inactivity_seconds: float)
        """
        now = time.time()
        bw = max(1.0, float(bbox[2] - bbox[0]))
        bh = max(1.0, float(bbox[3] - bbox[1]))
        face_scale = (bw + bh) / 2.0

        if track_id not in self.track_data:
            self.track_data[track_id] = {
                "last_landmarks": current_landmarks,
                "last_bbox": bbox,
                "last_move_time": now,
                "movement_history": [],
                "inactivity_seconds": 0.0
            }
            return (0.5, 0.0)

        td = self.track_data[track_id]
        prev_landmarks = td["last_landmarks"]

        # Compute average displacement
        disp_norm = 0.0
        if len(current_landmarks) > 0 and len(prev_landmarks) == len(current_landmarks):
            cur = np.array(current_landmarks)
            prev = np.array(prev_landmarks)
            dist = np.linalg.norm(cur - prev, axis=1)
            # Normalized by face scale
            disp_norm = float(np.mean(dist) / face_scale)
        else:
            # Fallback to bbox centroid displacement
            c_cur = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            p_box = td["last_bbox"]
            c_prev = ((p_box[0] + p_box[2]) / 2.0, (p_box[1] + p_box[3]) / 2.0)
            dist = np.hypot(c_cur[0] - c_prev[0], c_cur[1] - c_prev[1])
            disp_norm = float(dist / face_scale)

        # Jitter deadband: camera sensor noise is typically <0.012 of face scale
        deadband = 0.012
        if disp_norm < deadband:
            effective_disp = 0.0
        else:
            effective_disp = disp_norm - deadband

        # Scale displacement to normalized movement (0 to 1)
        raw_score = min(1.0, (effective_disp * 16.0) * self.sensitivity)
        
        # Smooth with recent history
        td["movement_history"].append(raw_score)
        if len(td["movement_history"]) > 10:
            td["movement_history"].pop(0)
        smoothed_movement = float(np.mean(td["movement_history"]))

        # Inactivity check: if smoothed movement is below 0.12 and raw score is small,
        # student is stationary (inactivity_seconds accumulates)
        if smoothed_movement > 0.12 or raw_score > 0.20:
            td["last_move_time"] = now
            td["inactivity_seconds"] = 0.0
        else:
            td["inactivity_seconds"] = max(0.0, now - td["last_move_time"])

        td["last_landmarks"] = current_landmarks
        td["last_bbox"] = bbox

        return (round(smoothed_movement, 3), round(td["inactivity_seconds"], 1))

    def cleanup(self, active_track_ids: List[int]):
        """Remove tracks that are no longer active."""
        current_keys = list(self.track_data.keys())
        for tid in current_keys:
            if tid not in active_track_ids:
                del self.track_data[tid]
