import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from config import Config

class HeadPoseEstimator:
    """
    Estimates 3D head pose (Yaw, Pitch, Roll) and granular student behavior:
    - Focused (looking at board / screen)
    - Sleeping / Head Down (pitch < -18° or head tilted down with prolonged stillness)
    - Drowsy / Inactive (high stillness duration)
    - Looking Down (reading, notebook, lap)
    - Looking Away (yaw > 22° left or right)
    - Looking Up
    Uses InsightFace 5-point facial landmarks with high-precision geometry,
    MediaPipe FaceMesh fallback, and PnP alignment.
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

        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=4,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.use_mediapipe = True
        except Exception as e:
            self.use_mediapipe = False

    def estimate_pose(
        self,
        frame_bgr: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        landmarks: Optional[Any] = None,
        inactivity_seconds: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculates yaw, pitch, roll and detailed behavioral state.
        """
        h, w = frame_bgr.shape[:2]
        center = (w / 2.0, h / 2.0)

        # 1. Primary High-Precision 5-Landmark Facial Geometry (from InsightFace)
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
                # When looking straight: ratio ~ 0.44
                # When looking down/sleeping: nose drops near mouth level -> ratio increases
                # When looking up: nose rises towards eyes -> ratio decreases
                nose_ratio = float((nose[1] - eye_mid[1]) / face_h)
                pitch = float(-(nose_ratio - 0.44) * 90.0)

                # Roll: eye horizon angle
                roll = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))

                # Behavioral state classification
                if pitch < -14.0 or (pitch < -8.0 and inactivity_seconds > 2.0) or (abs(roll) > 18.0 and inactivity_seconds > 2.0):
                    behavior = "Sleeping / Head Down"
                    is_looking = False
                elif inactivity_seconds > 4.0:
                    behavior = "Drowsy / Inactive"
                    is_looking = False
                elif pitch < -6.0:
                    behavior = "Looking Down"
                    is_looking = False
                elif abs(yaw) > Config.BOARD_YAW_MAX:
                    behavior = "Looking Away"
                    is_looking = False
                elif pitch > 16.0:
                    behavior = "Looking Up"
                    is_looking = False
                else:
                    behavior = "Focused"
                    is_looking = True

                return {
                    "yaw": round(yaw, 1),
                    "pitch": round(pitch, 1),
                    "roll": round(roll, 1),
                    "behavior": behavior,
                    "is_looking_at_board": is_looking,
                    "landmarks_2d": kps.tolist()
                }
            except Exception:
                pass

        # 2. MediaPipe FaceMesh PnP solver (if available)
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

                        if pitch < -14.0 or (pitch < -8.0 and inactivity_seconds > 2.0) or (abs(roll) > 18.0 and inactivity_seconds > 2.0):
                            behavior = "Sleeping / Head Down"
                        elif inactivity_seconds > 4.0:
                            behavior = "Drowsy / Inactive"
                        elif pitch < -6.0:
                            behavior = "Looking Down"
                        elif abs(yaw) > Config.BOARD_YAW_MAX:
                            behavior = "Looking Away"
                        elif pitch > 16.0:
                            behavior = "Looking Up"
                        else:
                            behavior = "Focused"

                        is_looking = (behavior == "Focused")

                        return {
                            "yaw": round(yaw, 1),
                            "pitch": round(pitch, 1),
                            "roll": round(roll, 1),
                            "behavior": behavior,
                            "is_looking_at_board": is_looking,
                            "landmarks_2d": image_points.tolist()
                        }
            except Exception:
                pass

        # 3. Geometric Bounding Box Fallback
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            fc_x = (x1 + x2) / 2.0
            fc_y = (y1 + y2) / 2.0
            approx_yaw = float(((fc_x - center[0]) / center[0]) * 35.0)
            approx_pitch = float(((fc_y - center[1]) / center[1]) * 20.0)

            if approx_pitch < -12.0 or (approx_pitch < -6.0 and inactivity_seconds > 2.0):
                behavior = "Sleeping / Head Down"
            elif inactivity_seconds > 4.0:
                behavior = "Drowsy / Inactive"
            elif approx_pitch < -5.0:
                behavior = "Looking Down"
            elif abs(approx_yaw) > Config.BOARD_YAW_MAX:
                behavior = "Looking Away"
            else:
                behavior = "Focused"

            is_looking = (behavior == "Focused")
            return {
                "yaw": round(approx_yaw, 1),
                "pitch": round(approx_pitch, 1),
                "roll": 0.0,
                "behavior": behavior,
                "is_looking_at_board": is_looking,
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
