import pytest
import numpy as np
from modules.vision.head_pose import HeadPoseEstimator
from modules.audio.capture import AudioStreamManager

def test_head_pose_focused():
    estimator = HeadPoseEstimator()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 5 landmarks: eyes level at y=150, mouth at y=250, nose centered at x=320, y=194 (ratio ~ 0.44)
    landmarks = [
        [280.0, 150.0],  # left eye
        [360.0, 150.0],  # right eye
        [320.0, 194.0],  # nose
        [290.0, 250.0],  # left mouth
        [350.0, 250.0],  # right mouth
    ]
    res = estimator.estimate_pose(dummy_frame, landmarks=landmarks, inactivity_seconds=0.0)
    assert res["behavior"] == "Focused"
    assert res["is_looking_at_board"] is True

def test_head_pose_sleeping_head_down():
    estimator = HeadPoseEstimator()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Nose drops down near mouth level -> looking down / sleeping posture
    landmarks = [
        [280.0, 150.0],
        [360.0, 150.0],
        [320.0, 230.0],  # Nose dropped towards mouth (ratio = 80/100 = 0.80 -> pitch = -(0.80 - 0.44)*90 = -32.4 deg)
        [290.0, 250.0],
        [350.0, 250.0],
    ]
    res = estimator.estimate_pose(dummy_frame, landmarks=landmarks, inactivity_seconds=1.0)
    assert res["behavior"] == "Sleeping / Head Down"
    assert res["is_looking_at_board"] is False

def test_head_pose_drowsy_inactivity():
    estimator = HeadPoseEstimator()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Neutral face geometry, but sitting motionless for > 5 seconds
    landmarks = [
        [280.0, 150.0],
        [360.0, 150.0],
        [320.0, 194.0],
        [290.0, 250.0],
        [350.0, 250.0],
    ]
    res = estimator.estimate_pose(dummy_frame, landmarks=landmarks, inactivity_seconds=6.0)
    assert res["behavior"] == "Drowsy / Inactive"
    assert res["is_looking_at_board"] is False

def test_head_pose_sleeping_side_tilt():
    estimator = HeadPoseEstimator()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Head tilted sideways (resting on shoulder/palm, roll > 20 deg) with inactivity > 2s
    landmarks = [
        [280.0, 130.0],  # left eye higher
        [360.0, 175.0],  # right eye lower -> roll ~ 29 deg
        [320.0, 195.0],
        [290.0, 240.0],
        [350.0, 275.0],
    ]
    res = estimator.estimate_pose(dummy_frame, landmarks=landmarks, inactivity_seconds=3.0)
    assert res["behavior"] == "Sleeping / Head Down"
    assert res["is_looking_at_board"] is False

def test_audio_metric_ingestion():
    audio_mgr = AudioStreamManager()
    
    # Ingest speech metric
    audio_mgr.ingest_client_metric(rms=0.015, is_speech=True)
    metrics = audio_mgr.get_latest_metrics()
    assert metrics["rms_loudness"] == 0.015
    assert metrics["speech_ratio"] > 0.0
    assert "Discussion" in metrics["classification"] or "Speech" in metrics["classification"]
