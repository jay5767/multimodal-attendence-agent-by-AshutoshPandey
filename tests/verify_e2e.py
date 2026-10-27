import sys
import requests
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_e2e_verification():
    base = "http://127.0.0.1:5000"
    print("=" * 60)
    print("  RUNNING FULL END-TO-END SYSTEM INTEGRATION TEST")
    print("=" * 60)

    # 1. System status check
    r = requests.get(f"{base}/api/system/status")
    assert r.status_code == 200
    sys_status = r.json()
    print(f"[✓] System Status Verified | Hardware: {sys_status['hardware']['provider']} | FPS: {sys_status['fps']}")

    # 2. Enroll Student 1: Ashutosh Pandey
    samples1 = [np.random.randn(512).tolist() for _ in range(5)]
    r1 = requests.post(f"{base}/api/students", json={
        "name": "Ashutosh Pandey",
        "roll_number": "0103CS211045",
        "samples": samples1
    })
    assert r1.status_code == 200
    print(f"[✓] Enrolled Ashutosh Pandey (0103CS211045)")

    # 3. Enroll Student 2: Priya Sharma
    samples2 = [np.random.randn(512).tolist() for _ in range(5)]
    r2 = requests.post(f"{base}/api/students", json={
        "name": "Priya Sharma",
        "roll_number": "0103CS211046",
        "samples": samples2
    })
    assert r2.status_code == 200
    print(f"[✓] Enrolled Priya Sharma (0103CS211046)")

    # 4. List all students
    r_list = requests.get(f"{base}/api/students")
    students = r_list.json()["students"]
    print(f"[✓] Total Enrolled Students: {len(students)}")
    assert len(students) >= 2

    # 5. Start Session
    r_start = requests.post(f"{base}/api/session/start", json={
        "name": "CS-401 Advanced Multimodal AI",
        "subject": "Computer Science"
    })
    assert r_start.status_code == 200
    sess = r_start.json()["session"]
    sess_id = sess["id"]
    print(f"[✓] Session #{sess_id} Started: '{sess['name']}' ({sess['subject']})")

    # 6. Verify Session is active
    r_stat = requests.get(f"{base}/api/session/status")
    assert r_stat.json()["active"] is True
    print(f"[✓] Verified Session Active Status: True")

    # 7. Stop Session
    r_stop = requests.post(f"{base}/api/session/stop")
    assert r_stop.status_code == 200
    print(f"[✓] Session #{sess_id} Stopped Successfully")

    # 8. Verify CSV Report Generation
    r_csv = requests.get(f"{base}/reports/session/{sess_id}/csv")
    assert r_csv.status_code == 200
    assert "Student Name" in r_csv.text
    print(f"[✓] CSV Attendance Register Generated: {len(r_csv.content)} bytes")

    # 9. Verify Printable HTML Report Generation
    r_html = requests.get(f"{base}/reports/session/{sess_id}/html")
    assert r_html.status_code == 200
    assert "Multimodal Attendance" in r_html.text
    assert "Ashutosh Pandey" in r_html.text
    print(f"[✓] Printable HTML Report Verified: {len(r_html.text)} characters")

    # 10. Verify Anonymous Mode Toggle
    r_anon = requests.post(f"{base}/api/privacy/anonymous-mode", json={"enabled": True})
    assert r_anon.status_code == 200
    r_students_anon = requests.get(f"{base}/api/students")
    first_student_anon = r_students_anon.json()["students"][0]
    assert "Student #" in first_student_anon["name"]
    print(f"[✓] Anonymous Mode Verified: Name masked as '{first_student_anon['name']}'")

    # Reset anonymous mode
    requests.post(f"{base}/api/privacy/anonymous-mode", json={"enabled": False})
    print(f"[✓] Anonymous Mode Restored to Normal")

    print("\n" + "=" * 60)
    print("  ALL 10 END-TO-END INTEGRATION CHECKS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    run_e2e_verification()
