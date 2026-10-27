"""Engagement signal analyzer aggregating frame-level visual observations into time windows."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import math
from config.settings import DEFAULT_CONFIG
from utils.logging_config import logger


@dataclass
class WindowedVisualMetrics:
    """Aggregated visual indicators for a single discrete time window."""
    window_start_sec: float
    window_end_sec: float
    active_participants_count: int
    avg_movement_speed: float
    forward_facing_ratio: float
    turned_away_ratio: float
    low_movement_duration_sec: float
    visual_activity_index: float  # 0.0 to 1.0
    data_quality: float  # 0.0 to 1.0


class EngagementAnalyzer:
    """Aggregates raw visual frame measurements into structured temporal windows."""

    def __init__(self, config=DEFAULT_CONFIG.engagement):
        self.config = config

    def aggregate_visual_windows(
        self,
        measurements: List[Dict[str, Any]],
        total_duration_sec: float,
        window_size_sec: int = DEFAULT_CONFIG.engagement.window_size_seconds
    ) -> List[WindowedVisualMetrics]:
        """Aggregate frame-level visual measurements into windowed summaries."""
        if total_duration_sec <= 0:
            total_duration_sec = 1.0

        num_windows = max(1, int(math.ceil(total_duration_sec / window_size_sec)))
        window_bins: List[List[Dict[str, Any]]] = [[] for _ in range(num_windows)]

        for m in measurements:
            t = float(m.get("timestamp_sec", 0.0))
            w_idx = min(num_windows - 1, max(0, int(t // window_size_sec)))
            window_bins[w_idx].append(m)

        results: List[WindowedVisualMetrics] = []

        for w_idx in range(num_windows):
            start_sec = float(w_idx * window_size_sec)
            end_sec = min(total_duration_sec, float((w_idx + 1) * window_size_sec))
            w_records = window_bins[w_idx]

            if not w_records:
                # No participants visible in this window
                results.append(WindowedVisualMetrics(
                    window_start_sec=round(start_sec, 2),
                    window_end_sec=round(end_sec, 2),
                    active_participants_count=0,
                    avg_movement_speed=0.0,
                    forward_facing_ratio=0.0,
                    turned_away_ratio=0.0,
                    low_movement_duration_sec=round(end_sec - start_sec, 2),
                    visual_activity_index=0.0,
                    data_quality=0.5  # Neutral quality when camera sees empty scene
                ))
                continue

            # Unique tracks in this window
            tracks = set(r["track_id"] for r in w_records if "track_id" in r)
            num_participants = len(tracks)

            # Movement speed statistics
            speeds = [
                float(r["head_movement_speed"]) for r in w_records
                if r.get("head_movement_speed") is not None
            ]
            avg_speed = float(np.mean(speeds)) if speeds else 0.0

            # Orientation states
            states = [r.get("observable_state", "unavailable") for r in w_records]
            valid_states = [s for s in states if s != "unavailable"]

            if valid_states:
                forward_cnt = sum(1 for s in valid_states if s == "forward_facing")
                turned_cnt = sum(1 for s in valid_states if s == "turned_away")
                forward_ratio = forward_cnt / len(valid_states)
                turned_ratio = turned_cnt / len(valid_states)
                data_quality = round(len(valid_states) / len(states), 2)
            else:
                forward_ratio = 0.0
                turned_ratio = 0.0
                data_quality = 0.2  # Low quality when states are mostly unavailable

            # Low movement duration estimate
            # Check frames where speed < low_movement_threshold
            low_speed_cnt = sum(1 for s in speeds if s < self.config.low_movement_threshold)
            low_ratio = (low_speed_cnt / len(speeds)) if speeds else 1.0
            low_movement_sec = (end_sec - start_sec) * low_ratio

            # Calculate composite Visual Activity Index (0.0 to 1.0)
            # Higher when there is moderate movement and orientation stability
            norm_speed = min(1.0, avg_speed / max(0.01, self.config.active_movement_threshold))
            # 60% weight on normalized movement, 40% on orientation presence
            activity_index = (0.50 * norm_speed) + (0.50 * forward_ratio)
            activity_index = round(float(np.clip(activity_index, 0.0, 1.0)), 3)

            results.append(WindowedVisualMetrics(
                window_start_sec=round(start_sec, 2),
                window_end_sec=round(end_sec, 2),
                active_participants_count=num_participants,
                avg_movement_speed=round(avg_speed, 3),
                forward_facing_ratio=round(forward_ratio, 3),
                turned_away_ratio=round(turned_ratio, 3),
                low_movement_duration_sec=round(low_movement_sec, 2),
                visual_activity_index=activity_index,
                data_quality=data_quality
            ))

        return results
