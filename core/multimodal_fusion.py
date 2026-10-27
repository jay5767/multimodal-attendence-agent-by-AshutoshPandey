"""Multimodal fusion engine combining synchronized visual and audio signals into session-level trends."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from config.settings import DEFAULT_CONFIG
from core.engagement_analyzer import WindowedVisualMetrics
from core.audio_analyzer import AudioIntervalMeasurement
from utils.logging_config import logger


@dataclass
class FusedWindowMetric:
    """Synchronized multimodal composite indicators for a discrete time window."""
    window_start_sec: float
    window_end_sec: float
    visual_activity_index: float
    speech_activity_index: float
    composite_activity_score: float
    active_participants_count: int
    data_quality_index: float
    signal_availability: str  # 'both', 'video_only', 'audio_only', 'none'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_start_sec": self.window_start_sec,
            "window_end_sec": self.window_end_sec,
            "visual_activity_index": round(self.visual_activity_index, 3),
            "speech_activity_index": round(self.speech_activity_index, 3),
            "composite_activity_score": round(self.composite_activity_score, 3),
            "active_participants_count": self.active_participants_count,
            "data_quality_index": round(self.data_quality_index, 2),
            "signal_availability": self.signal_availability
        }


class MultimodalFusionEngine:
    """Fuses synchronized visual observations and acoustic features using transparent weighted rules."""

    def __init__(self, fusion_config=DEFAULT_CONFIG.fusion):
        self.config = fusion_config

    def fuse_signals(
        self,
        visual_windows: List[WindowedVisualMetrics],
        audio_windows: List[AudioIntervalMeasurement],
        total_duration_sec: float,
        has_audio_track: bool
    ) -> List[FusedWindowMetric]:
        """Align visual and audio metrics on matching window timestamps."""
        fused_results: List[FusedWindowMetric] = []
        num_windows = max(len(visual_windows), len(audio_windows))

        if num_windows == 0:
            return fused_results

        # Index audio windows by approximate start second
        audio_by_start = {round(a.start_sec, 1): a for a in audio_windows}

        for v_win in visual_windows:
            start = round(v_win.window_start_sec, 1)
            end = round(v_win.window_end_sec, 1)

            # Find matching audio window
            a_win = audio_by_start.get(start)
            if a_win is None and audio_windows:
                # Find nearest audio window
                a_win = min(audio_windows, key=lambda a: abs(a.start_sec - start))

            v_idx = v_win.visual_activity_index
            v_quality = v_win.data_quality
            v_participants = v_win.active_participants_count

            if has_audio_track and a_win is not None:
                a_idx = a_win.speech_ratio
                availability = "both"
                # Data quality is composite of visual validity and acoustic completeness
                quality = (v_quality * 0.6) + 0.4
                # Weighted composite formula:
                # composite = (w_vis * visual_idx) + (w_aud * speech_ratio)
                composite = (self.config.weight_visual_movement * v_idx) + (self.config.weight_speech_activity * a_idx)
                # Rescale based on sum of active weights
                total_w = self.config.weight_visual_movement + self.config.weight_speech_activity
                composite = composite / max(0.01, total_w)
            else:
                a_idx = 0.0
                availability = "video_only"
                quality = v_quality * 0.8
                composite = v_idx  # 100% reliant on visual movement

            composite = float(np.clip(composite, 0.0, 1.0))

            fused_results.append(FusedWindowMetric(
                window_start_sec=v_win.window_start_sec,
                window_end_sec=v_win.window_end_sec,
                visual_activity_index=v_idx,
                speech_activity_index=a_idx,
                composite_activity_score=composite,
                active_participants_count=v_participants,
                data_quality_index=quality,
                signal_availability=availability
            ))

        return fused_results
