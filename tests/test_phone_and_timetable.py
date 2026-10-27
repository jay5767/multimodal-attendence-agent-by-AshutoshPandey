import pytest
import numpy as np
from modules.vision.phone_detector import PhoneDetector
from modules.timetable.schedule import timetable_manager, TIMETABLE_META, PERIOD_SLOTS

def test_phone_detector_initialization():
    detector = PhoneDetector()
    assert detector.is_loaded is True
    assert detector.model is not None

def test_phone_detector_inference():
    detector = PhoneDetector()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    phones = detector.detect_phones(dummy)
    assert isinstance(phones, list)

def test_phone_detector_match():
    detector = PhoneDetector()
    # Downward pitch into lap triggers posture phone usage
    is_phone, obj, reason = detector.match_phone_to_face([100, 100, 200, 200], [], pitch=-16.0)
    assert is_phone is True
    assert "Lap" in reason or "Phone" in reason

    # Upright board looking does not trigger phone usage
    is_phone, obj, reason = detector.match_phone_to_face([100, 100, 200, 200], [], pitch=2.0)
    assert is_phone is False

def test_timetable_metadata():
    assert "Lakshmi Narain" in TIMETABLE_META["college"]
    assert TIMETABLE_META["branch"] == "CSE (AIML)"
    assert TIMETABLE_META["section"] == "B"
    assert TIMETABLE_META["room"] == "Room No. 4-B9 / CV Raman Block"
    assert len(PERIOD_SLOTS) == 9

def test_timetable_current_and_simulation():
    # Simulate Monday Period II (11:30 AM -> AL-501 OS)
    period = timetable_manager.get_current_period(sim_day="MON", sim_time_str="11:30")
    assert period["is_active"] is True
    assert period["period"] == "II"
    assert period["subject_code"] == "AL-501"
    assert "Operating System" in period["subject_title"]
    assert "Dr. Aakriti Jain" in period["faculty"]

    # Simulate Lunch Period (01:45 PM -> Period V)
    lunch = timetable_manager.get_current_period(sim_day="MON", sim_time_str="13:45")
    assert lunch["is_active"] is True
    assert lunch["is_lunch"] is True
    assert lunch["period"] == "V"

def test_timetable_attention_recording():
    # Record live attention for AL-501
    timetable_manager.record_live_attention(
        "AL-501",
        attentive_students=20,
        distracted_students=4,
        phone_count=2,
        sleep_count=0,
        away_count=2
    )
    stats = timetable_manager.stats.get("AL-501")
    assert stats is not None
    assert stats["attention_rate"] > 0
    assert stats["distracted_phone"] >= 2
