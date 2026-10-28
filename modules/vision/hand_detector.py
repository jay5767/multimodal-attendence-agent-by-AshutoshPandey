"""
High-Precision Hand Raise Detector with Anti-Jitter Temporal Smoothing.
Detects elevated hand / arm in upper quadrants relative to tracked face.
"""

from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np

class HandRaiseDetector:
    def __init__(self):
        # track_id -> {"count": int, "active": bool, "cooldown": int}
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, track_id: int, frame_bgr: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> bool:
        """
        Evaluates whether a raised hand is present in the upper quadrant of the tracked face.
        Returns True if a sustained hand raise is confirmed, False otherwise.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False

        h, w = frame_bgr.shape[:2]
        fx1, fy1, fx2, fy2 = face_bbox
        fw = max(10, fx2 - fx1)
        fh = max(10, fy2 - fy1)

        state = self.tracks.setdefault(track_id, {
            "count": 0,
            "active": False,
            "cooldown": 0
        })

        # Raised hand search zones:
        # 1. Left of face (above chest level): [fx1 - 1.5*fw to fx1 - 0.1*fw, fy1 - 0.8*fh to fy1 + 0.4*fh]
        # 2. Right of face (above chest level): [fx2 + 0.1*fw to fx2 + 1.5*fw, fy1 - 0.8*fh to fy1 + 0.4*fh]
        # 3. Overhead zone: [fx1 - 0.3*fw to fx2 + 0.3*fw, fy1 - 1.2*fh to fy1]
        
        lz_x1 = max(0, int(fx1 - 1.5 * fw))
        lz_x2 = max(0, int(fx1 - 0.1 * fw))
        rz_x1 = min(w, int(fx2 + 0.1 * fw))
        rz_x2 = min(w, int(fx2 + 1.5 * fw))
        
        zone_y1 = max(0, int(fy1 - 0.9 * fh))
        zone_y2 = min(h, int(fy1 + 0.4 * fh))

        candidate = False

        if zone_y2 > zone_y1:
            ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            # Standard human skin tone mask in YCrCb
            skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
            # Mask out the face itself so facial skin doesn't count
            skin_mask[max(0, fy1):min(h, fy2), max(0, fx1):min(w, fx2)] = 0

            min_hand_pixels = int(0.18 * fw * fh)

            # Check left zone
            if lz_x2 > lz_x1:
                left_pixels = int(np.sum(skin_mask[zone_y1:zone_y2, lz_x1:lz_x2] > 0))
                if left_pixels >= min_hand_pixels:
                    candidate = True

            # Check right zone
            if not candidate and rz_x2 > rz_x1:
                right_pixels = int(np.sum(skin_mask[zone_y1:zone_y2, rz_x1:rz_x2] > 0))
                if right_pixels >= min_hand_pixels:
                    candidate = True

            # Check overhead zone
            if not candidate:
                tz_x1 = max(0, int(fx1 - 0.2 * fw))
                tz_x2 = min(w, int(fx2 + 0.2 * fw))
                tz_y1 = max(0, int(fy1 - 1.2 * fh))
                tz_y2 = max(0, int(fy1))
                if tz_x2 > tz_x1 and tz_y2 > tz_y1:
                    top_pixels = int(np.sum(skin_mask[tz_y1:tz_y2, tz_x1:tz_x2] > 0))
                    if top_pixels >= min_hand_pixels:
                        candidate = True

        # Temporal debouncing (requires 3 consecutive frames to trigger, holds for 12 frames)
        if candidate:
            state["count"] += 1
            if state["count"] >= 3:
                state["active"] = True
                state["cooldown"] = 12
        else:
            state["count"] = max(0, state["count"] - 1)
            if state["cooldown"] > 0:
                state["cooldown"] -= 1
            else:
                state["active"] = False

        return state["active"]

    def cleanup(self, active_track_ids):
        """Remove tracks that are no longer active."""
        current_keys = list(self.tracks.keys())
        for tid in current_keys:
            if tid not in active_track_ids:
                del self.tracks[tid]
