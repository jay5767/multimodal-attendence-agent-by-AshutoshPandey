import pytest
import cv2
import base64
import os
from app import app

def test_api_frame_process_endpoint():
    client = app.test_client()

    img_path = r"C:\Users\Ashutoshpandey\.gemini\antigravity\brain\911596d8-c5f1-43af-aa11-1f74c71aad6b\.user_uploaded\media_1793128556249_01b4a8db.png"
    assert os.path.exists(img_path)

    with open(img_path, "rb") as f:
        img_bytes = f.read()
    b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")

    res = client.post("/api/frame/process", json={
        "image": b64,
        "audio": {"rms": 0.045, "is_speech": True}
    })

    assert res.status_code == 200
    data = res.get_json()
    assert "faces" in data
    assert len(data["faces"]) >= 1
    assert "gauge" in data
    assert "metrics" in data
    assert data["metrics"]["attention"] >= 0.0
    print("Detected student in frame:", data["faces"][0]["name"])
