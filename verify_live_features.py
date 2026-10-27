import sys
import time
import requests
import socketio

BASE_URL = "http://localhost:5000"

with open("valid_test_b64.txt", "r") as f:
    VALID_B64 = "data:image/jpeg;base64," + f.read().strip()

def test_system_status():
    print("1. Testing system status endpoint...")
    res = requests.get(f"{BASE_URL}/api/system/status")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    print("   [OK] Hardware:", data.get("hardware"))
    print("   [OK] Audio devices found:", len(data.get("audio_devices", [])))

def test_socket_and_features():
    print("\n2. Testing WebSocket connection and Live Frame Ingestion...")
    sio = socketio.Client()
    received_frames = []
    received_audio_metrics = []

    @sio.on("connect")
    def on_connect():
        print("   [OK] WebSocket connected!")

    @sio.on("frame_processed")
    def on_frame(data):
        received_frames.append(data)
        metrics = data.get("metrics", {})
        print(f"   [FRAME EVENT] Gauge: {data.get('gauge')}% | Att: {metrics.get('attention')}% | Mov: {metrics.get('movement')}% | Inact: {metrics.get('inactivity')}s | Behavior: {metrics.get('behavior')}")

    sio.connect(BASE_URL, transports=['websocket'])
    time.sleep(1)

    # 3. Test Session Start
    print("\n3. Testing Session Start...")
    res = requests.post(f"{BASE_URL}/api/session/start", json={"name": "Verification Session", "subject": "CSE AIML"})
    assert res.status_code == 200
    sess = res.json()
    print("   [OK] Session started:", sess.get("session"))

    # 4. Generate synthetic frames:
    # Frame A: Upright synthetic face
    # We will test sending a base64 encoded test frame
    print("\n4. Sending test frames...")
    sio.emit("process_frame", {"image": TINY_JPEG_B64})
    time.sleep(1)

    # 5. Test Audio Ingestion
    print("\n5. Testing WebAudio Client Ingestion...")
    sio.emit("client_audio_metric", {"rms": 0.025, "is_speech": True})
    time.sleep(0.5)

    sio.emit("process_frame", {"image": TINY_JPEG_B64})
    time.sleep(1)

    assert len(received_frames) >= 1, "Should have received at least 1 processed frame!"
    last_frame = received_frames[-1]
    metrics = last_frame.get("metrics", {})
    print("   [OK] Verified metrics received:")
    print("        Attention:", metrics.get("attention"))
    print("        Movement:", metrics.get("movement"))
    print("        Inactivity:", metrics.get("inactivity"))
    print("        Speech Ratio:", metrics.get("speech_ratio"))
    print("        Behavior:", metrics.get("behavior"))
    print("        Gauge:", last_frame.get("gauge"))

    # 6. Test Instant Session Stop
    print("\n6. Testing Instant Session Stop (Must not freeze or lag)...")
    t0 = time.time()
    res = requests.post(f"{BASE_URL}/api/session/stop")
    dt = time.time() - t0
    print(f"   [STOP TIME] Stop completed in {dt:.3f} seconds!")
    assert dt < 2.0, f"Session stop took too long: {dt:.2f}s (must be < 2s)"
    assert res.status_code == 200
    stop_data = res.json()
    print("   [OK] Session stopped response:", stop_data.get("success"))

    # Verify session status is inactive
    res = requests.get(f"{BASE_URL}/api/session/status")
    status_data = res.json()
    print("   [OK] Session status active:", status_data.get("active"))
    assert status_data.get("active") is False

    sio.disconnect()
    print("\n[ALL TESTS PASSED SUCCESSFULLY!]", flush=True)
    os._exit(0)

if __name__ == "__main__":
    import os
    test_system_status()
    test_socket_and_features()
