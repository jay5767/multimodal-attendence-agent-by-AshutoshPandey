"""Multi-object person tracker using intersection-over-union (IoU) and spatial continuity.

Assigns consistent track IDs to observed individuals across classroom video frames.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from core.person_detector import DetectionResult
from config.settings import DEFAULT_CONFIG


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


@dataclass
class TrackedPerson:
    """Internal state of an actively or previously tracked individual."""
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    first_seen_sec: float
    last_seen_sec: float
    total_detections: int = 1
    missed_frames: int = 0
    matched_student_id: Optional[str] = None
    face_match_confidence: float = 0.0
    history_centers: List[Tuple[float, float, float]] = field(default_factory=list)  # (cx, cy, timestamp)

    @property
    def current_box(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def visible_duration_sec(self) -> float:
        return max(0.0, self.last_seen_sec - self.first_seen_sec)

    def update(self, det: DetectionResult, timestamp_sec: float) -> None:
        """Update track with a newly matched detection."""
        self.x1 = det.x1
        self.y1 = det.y1
        self.x2 = det.x2
        self.y2 = det.y2
        self.confidence = (self.confidence * 0.7) + (det.confidence * 0.3)
        self.last_seen_sec = timestamp_sec
        self.total_detections += 1
        self.missed_frames = 0
        cx, cy = self.center
        self.history_centers.append((cx, cy, timestamp_sec))
        if len(self.history_centers) > 30:
            self.history_centers.pop(0)

    def mark_missed(self) -> None:
        """Increment missed frame count when not detected."""
        self.missed_frames += 1


class PersonTracker:
    """IoU and centroid-based spatial tracker for multi-person video analysis."""

    def __init__(
        self,
        iou_threshold: float = DEFAULT_CONFIG.detection.iou_tracker_threshold,
        max_lost_frames: int = DEFAULT_CONFIG.detection.max_track_lost_frames,
        min_hits: int = DEFAULT_CONFIG.detection.min_track_hits
    ):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.min_hits = min_hits
        self.next_track_id = 1
        self.active_tracks: Dict[int, TrackedPerson] = {}
        self.all_completed_tracks: List[TrackedPerson] = []

    def update(self, detections: List[DetectionResult], timestamp_sec: float) -> List[TrackedPerson]:
        """Update existing tracks with detections or create new tracks."""
        if not self.active_tracks:
            # Initialize tracks for all current detections
            for det in detections:
                track = TrackedPerson(
                    track_id=self.next_track_id,
                    x1=det.x1,
                    y1=det.y1,
                    x2=det.x2,
                    y2=det.y2,
                    confidence=det.confidence,
                    first_seen_sec=timestamp_sec,
                    last_seen_sec=timestamp_sec,
                    total_detections=1,
                    history_centers=[(det.center[0], det.center[1], timestamp_sec)]
                )
                self.active_tracks[self.next_track_id] = track
                self.next_track_id += 1
            return list(self.active_tracks.values())

        track_ids = list(self.active_tracks.keys())
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = set(track_ids)

        if detections:
            # Build IoU cost matrix
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                t_box = self.active_tracks[tid].current_box
                for j, det in enumerate(detections):
                    d_box = (det.x1, det.y1, det.x2, det.y2)
                    iou_matrix[i, j] = compute_iou(t_box, d_box)

            # Greedy matching based on maximum IoU
            matched_pairs = []
            while True:
                max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0.0
                if max_iou < self.iou_threshold:
                    break
                row, col = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                matched_pairs.append((track_ids[row], col))
                # Invalidate row and col
                iou_matrix[row, :] = -1.0
                iou_matrix[:, col] = -1.0

            # Apply matched updates
            for tid, det_idx in matched_pairs:
                self.active_tracks[tid].update(detections[det_idx], timestamp_sec)
                unmatched_tracks.discard(tid)
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)

        # Handle unmatched active tracks
        for tid in unmatched_tracks:
            self.active_tracks[tid].mark_missed()

        # Archive dead tracks (missed > max_lost_frames)
        dead_ids = [tid for tid, track in self.active_tracks.items() if track.missed_frames > self.max_lost_frames]
        for tid in dead_ids:
            dead_track = self.active_tracks.pop(tid)
            if dead_track.total_detections >= self.min_hits:
                self.all_completed_tracks.append(dead_track)

        # Create new tracks for remaining unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_track = TrackedPerson(
                track_id=self.next_track_id,
                x1=det.x1,
                y1=det.y1,
                x2=det.x2,
                y2=det.y2,
                confidence=det.confidence,
                first_seen_sec=timestamp_sec,
                last_seen_sec=timestamp_sec,
                total_detections=1,
                history_centers=[(det.center[0], det.center[1], timestamp_sec)]
            )
            self.active_tracks[self.next_track_id] = new_track
            self.next_track_id += 1

        # Return currently active tracks
        return [t for t in self.active_tracks.values() if t.missed_frames == 0]

    def finalize(self) -> List[TrackedPerson]:
        """Call at end of video to retrieve all persistent tracks."""
        for track in self.active_tracks.values():
            if track.total_detections >= self.min_hits:
                self.all_completed_tracks.append(track)
        self.active_tracks.clear()
        return self.all_completed_tracks
