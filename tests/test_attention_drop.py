import pytest
from config import Config
from modules.attendance.manager import AttendanceManager
from modules.fusion.engine import MultimodalFusionEngine

@pytest.fixture
def fusion_engine(tmp_path):
    test_db = tmp_path / "test_drops.db"
    mgr = AttendanceManager(db_uri=f"sqlite:///{test_db}")
    return MultimodalFusionEngine(mgr)

def test_attention_drop_detection_triggered(fusion_engine):
    """
    Verifies that when the 3-interval moving average falls > 20% below
    the rolling session baseline, is_drop is flagged True with drop percentage.
    """
    fusion_engine.reset_session(start_epoch=0.0)
    fusion_engine.intervals_history = [
        {"interval_index": 1, "engagement": 85.0, "is_drop": False, "is_peak": False},
        {"interval_index": 2, "engagement": 55.0, "is_drop": False, "is_peak": False}
    ]
    fusion_engine.session_baseline_engagement = 80.0

    # Feed very low attention for Interval 3
    for _ in range(5):
        fusion_engine.feed_frame_metrics(
            attention_ratio=0.15,
            avg_movement=0.10,
            avg_inactivity_sec=28.0,
            audio_metrics={"speech_ratio": 0.0, "rms_loudness": 0.0}
        )

    # Step interval with elapsed > 30s
    interval_res = fusion_engine.step_interval_if_due(session_id=None, now_epoch=35.0)

    assert interval_res is not None
    # Current engagement is ~10-15%, moving average of [85, 55, ~12] is ~50.6
    # (80 - 50.6) / 80 = 29.4 / 80 = ~36.7% drop > 20% threshold!
    assert interval_res["is_drop"] is True
    assert interval_res["drop_percentage"] >= 20.0

def test_normal_fluctuation_does_not_trigger_drop(fusion_engine):
    """Verifies that minor normal fluctuations (< 20%) do not flag attention drops."""
    fusion_engine.reset_session(start_epoch=0.0)
    fusion_engine.intervals_history = [
        {"interval_index": 1, "engagement": 78.0, "is_drop": False, "is_peak": False},
        {"interval_index": 2, "engagement": 75.0, "is_drop": False, "is_peak": False}
    ]
    fusion_engine.session_baseline_engagement = 76.0

    # Feed steady engagement (around 72%)
    for _ in range(5):
        fusion_engine.feed_frame_metrics(
            attention_ratio=0.85,
            avg_movement=0.55,
            avg_inactivity_sec=4.0,
            audio_metrics={"speech_ratio": 0.25, "rms_loudness": 0.02}
        )

    interval_res = fusion_engine.step_interval_if_due(session_id=None, now_epoch=35.0)

    assert interval_res is not None
    assert interval_res["is_drop"] is False
    assert interval_res["drop_percentage"] < 20.0

def test_interaction_peak_detection(fusion_engine):
    """Verifies that high speech ratio and active movement trigger interaction peak."""
    fusion_engine.reset_session(start_epoch=0.0)
    fusion_engine.session_baseline_engagement = 65.0

    # Feed high discussion burst
    for _ in range(5):
        fusion_engine.feed_frame_metrics(
            attention_ratio=0.80,
            avg_movement=0.75,
            avg_inactivity_sec=1.0,
            audio_metrics={"speech_ratio": 0.85, "rms_loudness": 0.08}
        )

    interval_res = fusion_engine.step_interval_if_due(session_id=None, now_epoch=35.0)

    assert interval_res is not None
    assert interval_res["is_peak"] is True
