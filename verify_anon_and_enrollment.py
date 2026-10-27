import requests
import json
import numpy as np

BASE_URL = "http://127.0.0.1:5000"

def test_anonymous_mode():
    print("--- 1. Testing Anonymous Mode Toggle & Masking ---")
    
    # Check baseline status
    res = requests.get(f"{BASE_URL}/api/system/status")
    assert res.status_code == 200
    print("Baseline status:", res.json().get("anonymous_mode"))

    # Fetch students without anon
    res = requests.get(f"{BASE_URL}/api/students")
    data = res.json()
    assert data["success"] is True
    print("Non-anon students:", [(s["id"], s["name"], s["roll_number"]) for s in data["students"]])
    assert len(data["students"]) >= 1

    # Activate Anon Mode
    res = requests.post(f"{BASE_URL}/api/privacy/anonymous-mode", json={"enabled": True})
    assert res.status_code == 200
    assert res.json()["anonymous_mode"] is True
    print("Activated Anon Mode: OK")

    # Fetch students with anon ON
    res = requests.get(f"{BASE_URL}/api/students")
    data = res.json()
    assert data["success"] is True
    print("Anon students masked:", [(s["id"], s["name"], s["roll_number"]) for s in data["students"]])
    for s in data["students"]:
        assert "Student #" in s["name"], f"Expected masked name, got {s['name']}"
        assert "Roll #" in s["roll_number"], f"Expected masked roll, got {s['roll_number']}"

    # Fetch attendance with anon ON
    res = requests.get(f"{BASE_URL}/api/attendance/current")
    data = res.json()
    assert data["success"] is True
    for r in data.get("records", []):
        assert "Student #" in r["student_name"], f"Expected masked attendance name, got {r['student_name']}"
        assert "Roll #" in r["roll_number"], f"Expected masked roll, got {r['roll_number']}"
    print("Anon attendance records masked: OK")

    # Deactivate Anon Mode
    res = requests.post(f"{BASE_URL}/api/privacy/anonymous-mode", json={"enabled": False})
    assert res.status_code == 200
    assert res.json()["anonymous_mode"] is False
    print("Deactivated Anon Mode: OK")

    # Fetch students again - real names restored
    res = requests.get(f"{BASE_URL}/api/students")
    data = res.json()
    for s in data["students"]:
        assert "Student #" not in s["name"], f"Expected real name restored, got {s['name']}"
    print("Real student names restored: OK\n")

def test_enrollment_face_saving():
    print("--- 2. Testing Enrollment Face Data Saving ---")
    
    # Create 5 synthetic 512-d normalized face sample embeddings
    samples = []
    for _ in range(5):
        vec = np.random.randn(512).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        samples.append(vec.tolist())

    payload = {
        "name": "Ashutosh Pandey",
        "roll_number": "0103CS211045",
        "samples": samples
    }

    # Enroll student
    res = requests.post(f"{BASE_URL}/api/students", json=payload)
    print("Enrollment POST status:", res.status_code)
    data = res.json()
    print("Enrollment POST response:", data)
    assert res.status_code == 200
    assert data["success"] is True
    student_id = data["student"]["id"]

    # Verify student is in DB and active
    res = requests.get(f"{BASE_URL}/api/students")
    data = res.json()
    enrolled_ids = [s["id"] for s in data["students"]]
    assert student_id in enrolled_ids
    enrolled_names = [s["name"] for s in data["students"]]
    assert "Ashutosh Pandey" in enrolled_names
    print(f"Student #{student_id} ('Ashutosh Pandey') successfully enrolled and retrieved!")

    # Test re-enrolling/updating the same student roll number (must succeed gracefully without duplicate error)
    payload["name"] = "Ashutosh Pandey (Updated)"
    res = requests.post(f"{BASE_URL}/api/students", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["student"]["id"] == student_id
    print("Re-enrollment with existing roll number updated embeddings successfully!")

if __name__ == "__main__":
    test_anonymous_mode()
    test_enrollment_face_saving()
    print("\n[ALL ANONYMOUS AND ENROLLMENT TESTS PASSED!]")
