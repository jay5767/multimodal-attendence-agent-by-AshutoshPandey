import cv2
import numpy as np
from collections import deque, defaultdict
from typing import Dict, Any, Optional, Tuple, List
from config import Config

class HeadPoseEstimator:
    """
    Estimates 3D head pose (Yaw, Pitch, Roll) and stable student behavior:
    - Focused (looking at screen / board)
    - Sleeping / Head Down (pitch < -40° or sustained downward stillness)
    - Drowsy / Inactive (high stillness duration)
    - Looking Down (reading below desk)
    - Looking Away (|yaw| > 42°)
    - Looking Up (pitch > 36°)
    Applies temporal hysteresis debouncing to prevent sensitivity flicker.
    """
    def __init__(self):
        self.use_mediapipe = False
        self.face_mesh = None

        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        self.KEYPOINT_INDICES = [1, 152, 33, 263, 61, 291]
        self.mp_face_mesh = None

        # Debounce state history: {track_id: deque of recent (is_looking_raw, behavior_raw)}
        self._history = defaultdict(lambda: deque(maxlen=8))
        self._last_stable_state = {}

    def _debounce_state(self, track_id: int, is_looking_raw: bool, behavior_raw: str) -> Tuple[bool, str]:
        """
        Anti-jitter hysteresis: requires consecutive frames to switch state,
        preventing continuous up-and-down toggling.
        """
        buf = self._history[track_id]
        buf.append((is_looking_raw, behavior_raw))

        # Default initial state
        if track_id not in self._last_stable_state:
            self._last_stable_state[track_id] = (is_looking_raw, behavior_raw)
            return is_looking_raw, behavior_raw

        last_is_looking, last_behavior = self._last_stable_state[track_id]

        if last_is_looking:
            # Currently FOCUSED: require at least 5 of last 8 frames to be distracted before switching
            distracted_count = sum(1 for (looking, _) in buf if not looking)
            if len(buf) >= 6 and distracted_count >= 5:
                # Switch to distracted
                recent_behaviors = [b for (looking, b) in buf if not looking]
                chosen_b = recent_behaviors[-1] if recent_behaviors else behavior_raw
                self._last_stable_state[track_id] = (False, chosen_b)
                return False, chosen_b
            else:
                return True, "Focused"
        else:
            # Currently DISTRACTED: switch back to Focused promptly in 2 frames
            recent_3 = list(buf)[-3:] if len(buf) >= 3 else list(buf)
            focused_streak = sum(1 for (looking, _) in recent_3 if looking)
            if focused_streak >= 2:
                self._last_stable_state[track_id] = (True, "Focused")
                return True, "Focused"
            else:
                return False, last_behavior

    def estimate_pose(
        self,
        frame_bgr: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        landmarks: Optional[Any] = None,
        inactivity_seconds: float = 0.0,
        track_id: int = 1
    ) -> Dict[str, Any]:
        """
        Calculates yaw, pitch, roll and debounced behavioral state.
        """
        h, w = frame_bgr.shape[:2]
        center = (w / 2.0, h / 2.0)

        # 1. Primary High-Precision 5-Landmark Facial Geometry (from InsightFace or YuNet)
        if landmarks is not None and len(landmarks) >= 5:
            try:
                kps = np.array(landmarks, dtype=np.float32)
                le = kps[0]     # Left eye
                re = kps[1]     # Right eye
                nose = kps[2]   # Nose tip
                lm = kps[3]     # Left mouth corner
                rm = kps[4]     # Right mouth corner

                eye_mid = (le + re) / 2.0
                mouth_mid = (lm + rm) / 2.0
                face_h = max(1.0, float(mouth_mid[1] - eye_mid[1]))
                eye_dist = max(1.0, float(np.linalg.norm(re - le)))

                # Yaw: horizontal offset of nose from eye midpoint
                nose_dx = float(nose[0] - eye_mid[0])
                yaw = float((nose_dx / eye_dist) * 80.0)

                # Pitch: vertical offset of nose between eyes and mouth
                nose_ratio = float((nose[1] - eye_mid[1]) / face_h)
                pitch = float(-(nose_ratio - 0.54) * 80.0)

                # Roll: eye horizon angle
                roll = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))

                # Realistic classroom thresholds with generous screen-view margins:
                # Laptop/desktop screen interaction sits comfortably within pitch [-36 deg, +32 deg] and yaw [-42 deg, +42 deg].
                # Seated classroom/laptop interaction: screen viewing comfortably spans pitch [-18 deg, +20 deg] and yaw [-28 deg, +28 deg]
                # Sitting still while watching the screen is high focus attentiveness, not sleeping!
                if pitch < -24.0 or (pitch < -18.0 and inactivity_seconds > 8.0):
                    raw_behavior = "Sleeping / Head Down"
                    raw_looking = False
                elif inactivity_seconds > 45.0:
                    raw_behavior = "Drowsy / Inactive"
                    raw_looking = False
                elif abs(yaw) > 28.0:
                    raw_behavior = "Looking Away"
                    raw_looking = False
                elif pitch < -20.0:
                    raw_behavior = "Looking Down"
                    raw_looking = False
                elif pitch > 22.0:
                    raw_behavior = "Looking Up"
                    raw_looking = False
                else:
                    raw_behavior = "Focused"
                    raw_looking = True

                final_looking, final_behavior = self._debounce_state(track_id, raw_looking, raw_behavior)

                return {
                    "yaw": round(yaw, 1),
                    "pitch": round(pitch, 1),
                    "roll": round(roll, 1),
                    "behavior": final_behavior,
                    "is_looking_at_board": final_looking,
                    "landmarks_2d": kps.tolist()
                }
            except Exception:
                pass

        # 2. MediaPipe FaceMesh PnP solver (if active)
        if self.use_mediapipe and self.face_mesh is not None:
            try:
                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    target_landmarks = results.multi_face_landmarks[0]
                    focal_length = w
                    camera_matrix = np.array([
                        [focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]
                    ], dtype=np.float64)
                    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

                    image_points = []
                    for idx in self.KEYPOINT_INDICES:
                        lm_pt = target_landmarks.landmark[idx]
                        image_points.append((lm_pt.x * w, lm_pt.y * h))
                    image_points = np.array(image_points, dtype=np.float64)

                    success, rvec, tvec = cv2.solvePnP(
                        self.model_points,
                        image_points,
                        camera_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if success:
                        rmat, _ = cv2.Rodrigues(rvec)
                        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
                        if sy >= 1e-6:
                            pitch = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
                            yaw = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
                            roll = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
                        else:
                            pitch = float(np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1])))
                            yaw = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
                            roll = 0.0

                        yaw = ((yaw + 180) % 360) - 180
                        pitch = ((pitch + 180) % 360) - 180

                        if pitch < -38.0 or (pitch < -30.0 and inactivity_seconds > 6.0):
                            raw_behavior = "Sleeping / Head Down"
                            raw_looking = False
                        elif inactivity_seconds > 30.0:
                            raw_behavior = "Drowsy / Inactive"
                            raw_looking = False
                        elif abs(yaw) > 42.0:
                            raw_behavior = "Looking Away"
                            raw_looking = False
                        elif pitch < -34.0:
                            raw_behavior = "Looking Down"
                            raw_looking = False
                        elif pitch > 34.0:
                            raw_behavior = "Looking Up"
                            raw_looking = False
                        else:
                            raw_behavior = "Focused"
                            raw_looking = True

                        final_looking, final_behavior = self._debounce_state(track_id, raw_looking, raw_behavior)

                        return {
                            "yaw": round(yaw, 1),
                            "pitch": round(pitch, 1),
                            "roll": round(roll, 1),
                            "behavior": final_behavior,
                            "is_looking_at_board": final_looking,
                            "landmarks_2d": image_points.tolist()
                        }
            except Exception:
                pass

        # 3. Geometric Bounding Box Fallback (robust margins)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            fc_x = (x1 + x2) / 2.0
            fc_y = (y1 + y2) / 2.0
            approx_yaw = float(((fc_x - center[0]) / center[0]) * 35.0)
            approx_pitch = float(((fc_y - center[1]) / center[1]) * 20.0)

            if approx_pitch < -30.0 or (approx_pitch < -24.0 and inactivity_seconds > 15.0):
                raw_behavior = "Sleeping / Head Down"
                raw_looking = False
            elif inactivity_seconds > 45.0:
                raw_behavior = "Drowsy / Inactive"
                raw_looking = False
            elif abs(approx_yaw) > 42.0:
                raw_behavior = "Looking Away"
                raw_looking = False
            elif approx_pitch < -28.0:
                raw_behavior = "Looking Down"
                raw_looking = False
            else:
                raw_behavior = "Focused"
                raw_looking = True

            final_looking, final_behavior = self._debounce_state(track_id, raw_looking, raw_behavior)

            return {
                "yaw": round(approx_yaw, 1),
                "pitch": round(approx_pitch, 1),
                "roll": 0.0,
                "behavior": final_behavior,
                "is_looking_at_board": final_looking,
                "landmarks_2d": []
            }

        return {
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "behavior": "Focused",
            "is_looking_at_board": True,
            "landmarks_2d": []
        }
