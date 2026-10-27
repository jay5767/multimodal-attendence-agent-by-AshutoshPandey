"""Pedagogical AI recommendation agent based on explainable rule-based intelligence.

Translates empirical multimodal trends into actionable, advisory classroom insights.
Does NOT fabricate data or make consequential or punitive decisions about individuals.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from core.multimodal_fusion import FusedWindowMetric
from utils.timestamps import format_seconds_to_mmss
from utils.logging_config import logger


@dataclass
class TeachingRecommendation:
    """Actionable teaching recommendation with supporting evidence and limitations."""
    timestamp_start_sec: float
    timestamp_end_sec: float
    category: str  # 'Interaction Lull', 'Acoustic Surge', 'Movement Stillness', 'Pacing Opportunity'
    suggestion: str
    evidence: str
    confidence: float
    limitations: str
    status: str = "pending"  # 'pending', 'accepted', 'dismissed'
    instructor_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_start_sec": self.timestamp_start_sec,
            "timestamp_end_sec": self.timestamp_end_sec,
            "category": self.category,
            "suggestion": self.suggestion,
            "evidence": self.evidence,
            "confidence": round(self.confidence, 2),
            "limitations": self.limitations,
            "status": self.status,
            "instructor_notes": self.instructor_notes
        }


class RecommendationAgent:
    """Generates pedagogical suggestions from fused classroom activity signals."""

    def __init__(self):
        pass

    def generate_recommendations(
        self,
        fused_metrics: List[FusedWindowMetric],
        session_duration_sec: float,
        has_audio: bool
    ) -> List[TeachingRecommendation]:
        """Analyze temporal trends across windows and produce contextual suggestions."""
        recommendations: List[TeachingRecommendation] = []

        if not fused_metrics or len(fused_metrics) < 2:
            return recommendations

        # Adapt thresholds dynamically if session is a short benchmark/demo (< 60s)
        if session_duration_sec < 60.0:
            lull_min_duration = max(5.0, session_duration_sec * 0.30)
            stillness_min_duration = max(5.0, session_duration_sec * 0.35)
        else:
            lull_min_duration = 60.0
            stillness_min_duration = 90.0

        # 1. Detect sustained low speech / silence lulls
        if has_audio:
            silent_streak_start = None
            silent_streak_end = None

            for win in fused_metrics:
                if win.speech_activity_index < 0.12:
                    if silent_streak_start is None:
                        silent_streak_start = win.window_start_sec
                    silent_streak_end = win.window_end_sec
                else:
                    if silent_streak_start is not None and (silent_streak_end - silent_streak_start) >= lull_min_duration:
                        t_start = format_seconds_to_mmss(silent_streak_start)
                        t_end = format_seconds_to_mmss(silent_streak_end)
                        recommendations.append(TeachingRecommendation(
                            timestamp_start_sec=silent_streak_start,
                            timestamp_end_sec=silent_streak_end,
                            category="Interaction Lull",
                            suggestion="Consider inviting student questions or running a 1-minute think-pair-share check-in.",
                            evidence=f"Classroom acoustic speech ratio remained below 12% continuously from {t_start} to {t_end} (duration: {int(silent_streak_end - silent_streak_start)}s).",
                            confidence=0.85,
                            limitations="Silent intervals are normal during focused note-taking, reading, or quiet reflection and do not imply disengagement."
                        ))
                    silent_streak_start = None
                    silent_streak_end = None

            if silent_streak_start is not None and (silent_streak_end - silent_streak_start) >= lull_min_duration:
                t_start = format_seconds_to_mmss(silent_streak_start)
                t_end = format_seconds_to_mmss(silent_streak_end)
                recommendations.append(TeachingRecommendation(
                    timestamp_start_sec=silent_streak_start,
                    timestamp_end_sec=silent_streak_end,
                    category="Interaction Lull",
                    suggestion="Consider inviting student questions or running a 1-minute think-pair-share check-in.",
                    evidence=f"Classroom acoustic speech ratio remained below 12% continuously from {t_start} to {t_end}.",
                    confidence=0.85,
                    limitations="Silent intervals may reflect independent work, quizzes, or quiet problem solving."
                ))

        # 2. Detect sudden acoustic surges (potential collaborative buzz or classroom distraction)
        if has_audio:
            for i in range(1, len(fused_metrics)):
                prev = fused_metrics[i - 1]
                curr = fused_metrics[i]
                if curr.speech_activity_index > 0.60 and prev.speech_activity_index < 0.25:
                    t_start = format_seconds_to_mmss(curr.window_start_sec)
                    t_end = format_seconds_to_mmss(curr.window_end_sec)
                    recommendations.append(TeachingRecommendation(
                        timestamp_start_sec=curr.window_start_sec,
                        timestamp_end_sec=curr.window_end_sec,
                        category="Acoustic Surge",
                        suggestion="Check whether students are actively debating the course material or if a side conversation needs steering.",
                        evidence=f"Speech activity surged from {int(prev.speech_activity_index*100)}% to {int(curr.speech_activity_index*100)}% around {t_start}.",
                        confidence=0.80,
                        limitations="Audio metrics measure sound energy, not semantic dialogue content or speaker identity."
                    ))

        # 3. Detect sustained low movement / physical stillness
        still_streak_start = None
        still_streak_end = None
        for win in fused_metrics:
            if win.visual_activity_index < 0.20:
                if still_streak_start is None:
                    still_streak_start = win.window_start_sec
                still_streak_end = win.window_end_sec
            else:
                if still_streak_start is not None and (still_streak_end - still_streak_start) >= stillness_min_duration:
                    t_start = format_seconds_to_mmss(still_streak_start)
                    t_end = format_seconds_to_mmss(still_streak_end)
                    recommendations.append(TeachingRecommendation(
                        timestamp_start_sec=still_streak_start,
                        timestamp_end_sec=still_streak_end,
                        category="Movement Stillness",
                        suggestion="Consider a quick posture reset, live polling question, or switching presentation modalities.",
                        evidence=f"Visual observable movement remained minimal between {t_start} and {t_end} (duration: {int(still_streak_end - still_streak_start)}s).",
                        confidence=0.78,
                        limitations="Physical stillness often indicates deep concentration during instructional demonstrations."
                    ))
                still_streak_start = None
                still_streak_end = None

        if still_streak_start is not None and (still_streak_end - still_streak_start) >= stillness_min_duration:
            t_start = format_seconds_to_mmss(still_streak_start)
            t_end = format_seconds_to_mmss(still_streak_end)
            recommendations.append(TeachingRecommendation(
                timestamp_start_sec=still_streak_start,
                timestamp_end_sec=still_streak_end,
                category="Movement Stillness",
                suggestion="Consider offering a short stretch break or interactive discussion prompt.",
                evidence=f"Visual observable movement remained low continuously from {t_start} to {t_end}.",
                confidence=0.78,
                limitations="Low movement is common during silent reading or note-taking."
            ))

        # 4. Overall session pacing summary
        avg_composite = sum(w.composite_activity_score for w in fused_metrics) / len(fused_metrics)
        if avg_composite > 0.60:
            recommendations.append(TeachingRecommendation(
                timestamp_start_sec=0.0,
                timestamp_end_sec=session_duration_sec,
                category="Pacing Opportunity",
                suggestion="Overall session exhibited consistent observable activity. Great momentum for high-interaction seminar topics.",
                evidence=f"Average composite activity score was {round(avg_composite, 2)} across {len(fused_metrics)} analyzed time windows.",
                confidence=0.90,
                limitations="Aggregate indicators provide broad retrospective trends rather than granular individual evaluations."
            ))
        elif avg_composite < 0.25:
            recommendations.append(TeachingRecommendation(
                timestamp_start_sec=0.0,
                timestamp_end_sec=session_duration_sec,
                category="Pacing Opportunity",
                suggestion="Session followed a lecture-heavy pattern. Adding structured breaks or student checkpoints may enhance participation.",
                evidence=f"Average composite activity index was {round(avg_composite, 2)} with limited measured movement/speech variations.",
                confidence=0.85,
                limitations="Formal lectures naturally produce lower observable movement and speech signals."
            ))

        return recommendations
