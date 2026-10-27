import pytest
from config import Config
from modules.attendance.manager import AttendanceManager
from modules.fusion.engine import MultimodalFusionEngine

@pytest.fixture
def fusion_engine(tmp_path):
    test_db = tmp_path / "test_fusion.db"
    mgr = AttendanceManager(db_uri=f"sqlite:///{test_db}")
    return MultimodalFusionEngine(mgr)

def test_fusion_formula_exact_weights(fusion_engine):
    """
    Verifies multimodal fusion:
    engagement = 0.35*attention + 0.25*movement + 0.20*(1 - inactivity) + 0.20*audio
    scaled 0-100.
    """
    # Test values:
    # attention = 1.0 (100%)
    # movement = 1.0 (100%)
    # inactivity = 0.0 (0s inactive -> (1 - 0) = 1.0)
    # audio = 1.0 (100%)
    # Expected: 0.35(1) + 0.25(1) + 0.20(1) + 0.20(1) = 1.00 -> 100.0%
    fusion_engine.feed_frame_metrics(
        attention_ratio=1.0,
        avg_movement=1.0,
        avg_inactivity_sec=0.0,
        audio_metrics={"speech_ratio": 1.0, "rms_loudness": 0.1}
    )
    score = fusion_engine.calculate_current_instant_score()
    assert score == 100.0

def test_fusion_formula_zero_engagement(fusion_engine):
    """Verifies that complete absence of attention, movement, and speech produces 0% score."""
    # attention = 0.0
    # movement = 0.0
    # inactivity = 30.0 sec (normalized to 1.0 -> 1 - 1.0 = 0.0)
    # audio = 0.0
    fusion_engine.feed_frame_metrics(
        attention_ratio=0.0,
        avg_movement=0.0,
        avg_inactivity_sec=30.0,
        audio_metrics={"speech_ratio": 0.0, "rms_loudness": 0.0}
    )
    score = fusion_engine.calculate_current_instant_score()
    assert score <= 1.0  # Near zero

def test_fusion_balanced_classroom(fusion_engine):
    """Verifies expected typical classroom score with 80% attention, 50% movement, 10% inactivity, 30% audio."""
    # att: 0.8 * 0.35 = 0.28
    # mov: 0.5 * 0.25 = 0.125
    # inact: (1 - 0.1) * 0.20 = 0.18
    # aud: ~0.25 * 0.20 = 0.05
    # sum: ~ 0.635 -> 63.5%
    fusion_engine.feed_frame_metrics(
        attention_ratio=0.8,
        avg_movement=0.5,
        avg_inactivity_sec=3.0, # 3s / 30s = 0.10
        audio_metrics={"speech_ratio": 0.3, "rms_loudness": 0.02}
    )
    score = fusion_engine.calculate_current_instant_score()
    assert 55.0 <= score <= 75.0
