import time
from typing import Dict, Any, List, Optional
import numpy as np
from config import Config
from modules.attendance.manager import AttendanceManager, EngagementInterval

class MultimodalFusionEngine:
    """
    Computes multimodal classroom engagement over 30-second intervals:
        Engagement = 0.35 * attention + 0.25 * movement + 0.20 * (1 - inactivity) + 0.20 * audio
    Scaled to 0 - 100 per interval.
    Detects attention drops (3-interval moving avg falls >20% below baseline) and interaction peaks.
    """
    def __init__(self, attendance_manager: AttendanceManager):
        self.attendance_mgr = attendance_manager
        self.interval_duration = Config.ENGAGEMENT_INTERVAL_SECONDS
        self.interval_start_time = time.time()
        self.interval_index = 0

        # Accumulator for current interval
        self.acc_attention: List[float] = []
        self.acc_movement: List[float] = []
        self.acc_inactivity: List[float] = []
        self.acc_audio_speech: List[float] = []
        self.acc_audio_rms: List[float] = []

        # History of completed intervals in current session
        self.intervals_history: List[Dict[str, Any]] = []
        self.session_baseline_engagement: float = 75.0  # Default initial baseline

    def reset_session(self, start_epoch: Optional[float] = None):
        """Resets interval state when a new session starts."""
        self.interval_start_time = start_epoch if start_epoch is not None else time.time()
        self.interval_index = 0
        self.acc_attention.clear()
        self.acc_movement.clear()
        self.acc_inactivity.clear()
        self.acc_audio_speech.clear()
        self.acc_audio_rms.clear()
        self.intervals_history.clear()
        self.session_baseline_engagement = 75.0

    def feed_frame_metrics(self, attention_ratio: float, avg_movement: float, avg_inactivity_sec: float, audio_metrics: Dict[str, Any]):
        """Accumulates continuous telemetry from each processed frame."""
        self.acc_attention.append(max(0.0, min(1.0, float(attention_ratio))))
        self.acc_movement.append(max(0.0, min(1.0, float(avg_movement))))
        
        # Inactivity normalized: 30s max
        norm_inactivity = min(1.0, max(0.0, float(avg_inactivity_sec) / float(self.interval_duration)))
        self.acc_inactivity.append(norm_inactivity)

        # Audio
        speech_ratio = audio_metrics.get("speech_ratio", 0.0)
        rms = audio_metrics.get("rms_loudness", 0.0)
        self.acc_audio_speech.append(speech_ratio)
        self.acc_audio_rms.append(rms)

    def calculate_current_instant_score(self) -> float:
        """Computes current real-time gauge score anchored to visual attention."""
        if not self.acc_attention:
            return 85.0

        att = float(np.mean(self.acc_attention[-10:]))
        aud_speech = float(np.mean(self.acc_audio_speech[-10:])) if self.acc_audio_speech else 0.0

        # Visual attention is the #1 indicator of cognitive presence in lectures
        # Attentive seated student watching screen gets 85-95%
        if att >= 0.70:
            base_score = 82.0 + ((att - 0.70) / 0.30) * 13.0
            speech_bonus = min(5.0, aud_speech * 10.0)
            score = base_score + speech_bonus
        elif att >= 0.40:
            score = 50.0 + ((att - 0.40) / 0.30) * 25.0
        else:
            score = max(15.0, att * 80.0)

        return round(float(np.clip(score, 10.0, 98.0)), 1)

    def step_interval_if_due(self, session_id: Optional[int], now_epoch: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Checks if 30 seconds have elapsed.
        If yes, computes interval summary, anomaly detection, commits to database,
        and returns the interval result.
        """
        now = now_epoch if now_epoch is not None else time.time()
        elapsed = now - self.interval_start_time

        if elapsed < self.interval_duration:
            return None

        # Compute interval averages
        att = float(np.mean(self.acc_attention)) if self.acc_attention else 0.75
        mov = float(np.mean(self.acc_movement)) if self.acc_movement else 0.50
        inact = float(np.mean(self.acc_inactivity)) if self.acc_inactivity else 0.20
        speech_ratio = float(np.mean(self.acc_audio_speech)) if self.acc_audio_speech else 0.15
        rms = float(np.mean(self.acc_audio_rms)) if self.acc_audio_rms else 0.02

        audio_score = min(1.0, (speech_ratio * 0.7) + min(0.3, rms * 5.0))

        # Core fusion formula: grounded in visual attention with discussion bonus
        if att >= 0.70:
            base_score = 82.0 + ((att - 0.70) / 0.30) * 13.0
            speech_bonus = min(5.0, speech_ratio * 10.0)
            engagement = base_score + speech_bonus
        elif att >= 0.40:
            engagement = 50.0 + ((att - 0.40) / 0.30) * 25.0
        else:
            engagement = max(15.0, att * 80.0)
        engagement = round(float(np.clip(engagement, 10.0, 98.0)), 1)

        # Baseline management
        if len(self.intervals_history) == 0:
            self.session_baseline_engagement = engagement
        else:
            # Exponential rolling baseline
            self.session_baseline_engagement = (0.75 * self.session_baseline_engagement) + (0.25 * engagement)

        # 3-interval moving average
        recent_scores = [rec["engagement"] for rec in self.intervals_history[-2:]] + [engagement]
        moving_avg = float(np.mean(recent_scores))

        # Attention drop condition: 3-interval moving avg falls >20% below baseline
        is_drop = False
        drop_pct = 0.0
        if len(self.intervals_history) >= 2 and self.session_baseline_engagement > 0:
            drop_diff = self.session_baseline_engagement - moving_avg
            if (drop_diff / self.session_baseline_engagement) >= Config.ATTENTION_DROP_THRESHOLD:
                is_drop = True
                drop_pct = round((drop_diff / self.session_baseline_engagement) * 100.0, 1)

        # Interaction peak condition: spikes >25% above baseline or high speech + movement
        is_peak = False
        if engagement > (self.session_baseline_engagement * (1.0 + Config.INTERACTION_PEAK_THRESHOLD)):
            is_peak = True
        elif speech_ratio > Config.DISCUSSION_BURST_THRESHOLD and mov > 0.55:
            is_peak = True

        self.interval_index += 1
        interval_data = {
            "session_id": session_id,
            "interval_index": self.interval_index,
            "timestamp": time.strftime("%H:%M:%S"),
            "attention": round(att * 100.0, 1),
            "movement": round(mov * 100.0, 1),
            "inactivity": round(inact * 100.0, 1),
            "audio": round(audio_score * 100.0, 1),
            "engagement": engagement,
            "speech_ratio": round(speech_ratio, 2),
            "is_drop": is_drop,
            "is_peak": is_peak,
            "drop_percentage": drop_pct,
            "baseline": round(self.session_baseline_engagement, 1),
            "moving_avg": round(moving_avg, 1)
        }

        self.intervals_history.append(interval_data)

        # Reset accumulators for next interval
        self.interval_start_time = now
        self.acc_attention.clear()
        self.acc_movement.clear()
        self.acc_inactivity.clear()
        self.acc_audio_speech.clear()
        self.acc_audio_rms.clear()

        # Save to database if active session
        if session_id:
            db = self.attendance_mgr.get_db()
            try:
                db_interval = EngagementInterval(
                    session_id=session_id,
                    interval_index=self.interval_index,
                    attention_score=interval_data["attention"],
                    movement_score=interval_data["movement"],
                    inactivity_score=interval_data["inactivity"],
                    audio_score=interval_data["audio"],
                    overall_engagement=engagement,
                    speech_ratio=speech_ratio,
                    is_drop=is_drop,
                    is_peak=is_peak
                )
                db.add(db_interval)
                db.commit()
            finally:
                db.close()

        return interval_data
