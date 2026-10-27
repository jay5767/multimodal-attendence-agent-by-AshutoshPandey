import time
import pytest
from config import Config
from modules.attendance.manager import AttendanceManager, Student, Session, AttendanceRecord

@pytest.fixture
def manager(tmp_path):
    test_db = tmp_path / "test_attendance.db"
    mgr = AttendanceManager(db_uri=f"sqlite:///{test_db}")
    
    # Create two test students
    db = mgr.get_db()
    s1 = Student(name="Ashutosh Pandey", roll_number="0103CS211045", embedding_json="[]", is_active=True)
    s2 = Student(name="Priya Sharma", roll_number="0103CS211046", embedding_json="[]", is_active=True)
    db.add_all([s1, s2])
    db.commit()
    db.close()
    return mgr

def test_sustained_presence_verified_at_70_percent(manager):
    """Verifies that a student seen in >= 70% of frames in rolling window transitions to Present."""
    sess = manager.start_session(name="Test Session", subject="AI")
    session_id = sess["id"]
    t0 = time.time()

    # Simulate 10 frames over 20 seconds where student 1 is present in 8 out of 10 (80% >= 70%)
    for i in range(10):
        t_frame = t0 + (i * 2.0)
        # Student 1 detected in frames 0..7 (8 frames)
        detected = [1] if i < 8 else []
        records = manager.process_frame(detected, now_epoch=t_frame)

    db = manager.get_db()
    rec1 = db.query(AttendanceRecord).filter_by(session_id=session_id, student_id=1).first()
    rec2 = db.query(AttendanceRecord).filter_by(session_id=session_id, student_id=2).first()

    assert rec1.sustained_verified is True
    assert rec1.status == "Present"
    assert rec1.total_presence_seconds > 0

    # Student 2 was never seen
    assert rec2.sustained_verified is False
    assert rec2.status == "Absent"
    db.close()

def test_sustained_presence_rejected_below_70_percent(manager):
    """Verifies that a student seen in only ~33% of frames (< 70%) is not marked Present."""
    sess = manager.start_session(name="Low Presence Session", subject="Math")
    session_id = sess["id"]
    t0 = time.time()

    # Simulate 12 frames where student 1 is seen every 3rd frame (33%)
    for i in range(12):
        t_frame = t0 + (i * 2.0)
        detected = [1] if (i % 3 == 0) else []
        manager.process_frame(detected, now_epoch=t_frame)

    db = manager.get_db()
    rec1 = db.query(AttendanceRecord).filter_by(session_id=session_id, student_id=1).first()
    assert rec1.sustained_verified is False
    # Not Present (either Partial or Absent)
    assert rec1.status in ("Partial", "Absent")
    db.close()

def test_late_marking_after_grace_period(manager):
    """Verifies that a student first verified after the late grace period (5 min) is marked Late."""
    sess = manager.start_session(name="Late Session", subject="Physics")
    session_id = sess["id"]
    t0 = time.time()

    # First presence occurs at minute 6 (360 seconds after session start > 300s grace period)
    late_start = t0 + 360.0
    for i in range(10):
        t_frame = late_start + (i * 1.5)
        manager.process_frame([1], now_epoch=t_frame)

    db = manager.get_db()
    rec1 = db.query(AttendanceRecord).filter_by(session_id=session_id, student_id=1).first()
    assert rec1.sustained_verified is True
    assert rec1.status == "Late"
    db.close()

def test_session_end_marks_unseen_students_absent(manager):
    """Verifies that unseen enrolled students are auto-marked Absent upon session completion."""
    sess = manager.start_session(name="Closing Session", subject="AI")
    ended = manager.end_session()

    assert ended is not None
    assert ended["is_active"] is False

    db = manager.get_db()
    records = db.query(AttendanceRecord).filter_by(session_id=sess["id"]).all()
    for r in records:
        assert r.status == "Absent"
    db.close()
