import math
from typing import Dict, List, Optional, Tuple
import numpy as np

class TrackedFace:
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int]):
        self.track_id = track_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.centroid = self._calc_centroid(bbox)
        self.disappeared_frames = 0
        self.student_id: Optional[int] = None
        self.student_name: str = "Unknown"
        self.similarity: float = 0.0
        self.yaw: float = 0.0
        self.pitch: float = 0.0
        self.roll: float = 0.0
        self.is_looking_at_board: bool = True
        self.movement_score: float = 0.0
        self.inactivity_seconds: float = 0.0
        self.history_centroids: List[Tuple[float, float]] = [self.centroid]

    def _calc_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, bbox: Tuple[int, int, int, int]):
        self.bbox = bbox
        self.centroid = self._calc_centroid(bbox)
        self.disappeared_frames = 0
        self.history_centroids.append(self.centroid)
        if len(self.history_centroids) > 30:
            self.history_centroids.pop(0)

class FaceTracker:
    def __init__(self, max_disappeared: int = 25, iou_threshold: float = 0.3):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedFace] = {}

    def _compute_iou(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_width = max(0, xB - xA)
        inter_height = max(0, yB - yA)
        inter_area = inter_width * inter_height

        boxA_area = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxB_area = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        union = float(boxA_area + boxB_area - inter_area)
        if union <= 0:
            return 0.0
        return inter_area / union

    def _centroid_distance(self, c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[TrackedFace]:
        """
        detections: list of bboxes [(x1, y1, x2, y2), ...]
        Returns active tracked faces.
        """
        if len(detections) == 0:
            # Mark all existing as disappeared
            for tid in list(self.tracks.keys()):
                self.tracks[tid].disappeared_frames += 1
                if self.tracks[tid].disappeared_frames > self.max_disappeared:
                    del self.tracks[tid]
            return list(self.tracks.values())

        if len(self.tracks) == 0:
            for bbox in detections:
                tf = TrackedFace(self.next_track_id, bbox)
                self.tracks[self.next_track_id] = tf
                self.next_track_id += 1
            return list(self.tracks.values())

        # Match existing tracks with detections via IoU + Centroid proximity
        track_ids = list(self.tracks.keys())
        matched_detections = set()
        matched_tracks = set()

        # Step 1: IoU matrix matching
        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            for j, det_box in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(self.tracks[tid].bbox, det_box)

        # Greedy match on highest IoU >= threshold
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            i, j = idx[0], idx[1]
            tid = track_ids[i]
            self.tracks[tid].update(detections[j])
            matched_tracks.add(tid)
            matched_detections.add(j)
            iou_matrix[i, :] = -1.0
            iou_matrix[:, j] = -1.0

        # Step 2: For unmatched detections, try centroid distance
        unmatched_dets = [j for j in range(len(detections)) if j not in matched_detections]
        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]

        for j in unmatched_dets:
            det_box = detections[j]
            det_centroid = ((det_box[0] + det_box[2]) / 2.0, (det_box[1] + det_box[3]) / 2.0)
            best_tid = None
            best_dist = 120.0  # max 120 pixels movement between frames
            for tid in unmatched_tracks:
                dist = self._centroid_distance(self.tracks[tid].centroid, det_centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                self.tracks[best_tid].update(det_box)
                unmatched_tracks.remove(best_tid)
                matched_detections.add(j)
            else:
                # New track
                tf = TrackedFace(self.next_track_id, det_box)
                self.tracks[self.next_track_id] = tf
                self.next_track_id += 1

        # Step 3: Handle disappeared tracks
        for tid in unmatched_tracks:
            self.tracks[tid].disappeared_frames += 1
            if self.tracks[tid].disappeared_frames > self.max_disappeared:
                del self.tracks[tid]

        return [t for t in self.tracks.values() if t.disappeared_frames == 0]
