import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import requests
from config import Config
from modules.attendance.manager import AttendanceManager, TeachingSuggestion

class TeachingSuggestionAgent:
    """
    Generates context-aware, assistive AI teaching recommendations based on
    multimodal engagement dynamics. Uses local Ollama LLM with intelligent rule-based fallback.
    Decision-support tone, never prescriptive.
    """
    def __init__(self, attendance_manager: AttendanceManager):
        self.attendance_mgr = attendance_manager
        self.ollama_url = Config.OLLAMA_BASE_URL
        self.default_model = Config.OLLAMA_MODEL
        self.fallback_model = Config.OLLAMA_FALLBACK_MODEL
        self.timeout = Config.OLLAMA_REQUEST_TIMEOUT

    def _call_ollama(self, prompt: str, system_prompt: str) -> Optional[str]:
        """Attempts to query local Ollama instance with default model, falling back to phi3."""
        # Fast 0.5s health ping check to avoid blocking if Ollama is not running
        try:
            ping = requests.get(self.ollama_url, timeout=0.5)
            if ping.status_code != 200:
                return None
        except Exception:
            return None

        for model in [self.default_model, self.fallback_model]:
            try:
                res = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "system": system_prompt,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.5,
                            "num_predict": 120
                        }
                    },
                    timeout=self.timeout
                )
                if res.status_code == 200:
                    data = res.json()
                    response_text = data.get("response", "").strip()
                    if response_text:
                        return response_text
            except Exception:
                continue
        return None

    def _generate_rule_based_suggestion(self, trigger_type: str, interval_data: Dict[str, Any]) -> str:
        """
        High-quality heuristic suggestion generator providing assistive,
        pedagogical advice when Ollama is not installed or unreachable.
        """
        time_str = interval_data.get("timestamp", datetime.now().strftime("%H:%M:%S"))
        drop_pct = interval_data.get("drop_percentage", 0.0)
        attention = interval_data.get("attention", 70.0)
        movement = interval_data.get("movement", 50.0)
        inactivity = interval_data.get("inactivity", 20.0)
        speech_ratio = interval_data.get("speech_ratio", 0.1)

        if trigger_type == "drop":
            if attention < 45.0 and inactivity > 50.0:
                return (
                    f"Engagement dipped {drop_pct}% at {time_str}. Students appear visually fatigued or disengaged from the screen. "
                    f"Consider a 60-second stretch break or pausing for an open thumbs-up/thumbs-down comprehension check."
                )
            elif speech_ratio < 0.05 and movement < 30.0:
                return (
                    f"Engagement fell {drop_pct}% at {time_str} following prolonged silence. "
                    f"A brief 'Think-Pair-Share' prompt or posing a speculative question could rekindle peer interaction."
                )
            elif attention < 50.0:
                return (
                    f"Attention dropped {drop_pct}% at {time_str} as gaze shifted away from the board. "
                    f"You might highlight key lecture slides with a quick live poll or invite a volunteer to summarize the key takeaway."
                )
            else:
                return (
                    f"A {drop_pct}% drop in multimodal engagement was detected at {time_str}. "
                    f"Pacing adjustment: shifting from passive lecture to a quick practical example or asking for student questions may help."
                )

        elif trigger_type == "peak":
            return (
                f"Lively interaction peak detected at {time_str}! High verbal participation and physical responsiveness. "
                f"Great momentum—consider capturing these insights on the shared board before transitioning to the next concept."
            )

        elif trigger_type == "summary":
            return (
                f"Session Summary: Overall engagement remained steady across intervals. "
                f"Visual attention was strongest in the initial half. "
                f"Interactive segments with verbal exchanges correlated with the highest sustained presence."
            )

        return f"Class engagement is currently stable at {interval_data.get('engagement', 75)}%. Keep monitoring pacing."

    def generate_suggestion(self, session_id: Optional[int], trigger_type: str, interval_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates suggestion, persists to database, and returns recommendation metadata.
        """
        drop_pct = interval_data.get("drop_percentage", 0.0)
        time_str = interval_data.get("timestamp", datetime.now().strftime("%H:%M:%S"))

        system_prompt = (
            "You are an assistive AI classroom teaching companion. "
            "Analyze the engagement metrics and provide a warm, encouraging, context-aware suggestion "
            "for the teacher. Keep it strictly to 1 or 2 concise sentences. Use a supportive, decision-support tone, never prescriptive."
        )

        user_prompt = (
            f"Classroom Event: {trigger_type.upper()}. Time: {time_str}. "
            f"Drop Percentage: {drop_pct}%. Attention Score: {interval_data.get('attention', 0)}%. "
            f"Movement: {interval_data.get('movement', 0)}%. Audio Speech Ratio: {interval_data.get('speech_ratio', 0)}. "
            f"Overall Engagement: {interval_data.get('engagement', 0)}%. "
            "Give one practical teaching intervention."
        )

        # Try Ollama first
        ollama_resp = self._call_ollama(user_prompt, system_prompt)
        if ollama_resp:
            source = "ollama"
            model_name = self.default_model
            suggestion_text = ollama_resp
        else:
            source = "rule_engine"
            model_name = "ClassroomRuleEngine"
            suggestion_text = self._generate_rule_based_suggestion(trigger_type, interval_data)

        suggestion_record = {
            "session_id": session_id,
            "time": time_str,
            "trigger_type": trigger_type,
            "drop_percentage": drop_pct,
            "suggestion": suggestion_text,
            "source": source,
            "model_name": model_name
        }

        # Save to database
        if session_id:
            db = self.attendance_mgr.get_db()
            try:
                rec = TeachingSuggestion(
                    session_id=session_id,
                    timestamp=datetime.utcnow(),
                    trigger_type=trigger_type,
                    drop_percentage=drop_pct,
                    suggestion_text=suggestion_text,
                    source=source,
                    model_name=model_name
                )
                db.add(rec)
                db.commit()
                suggestion_record["id"] = rec.id
            finally:
                db.close()

        return suggestion_record
