import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from config import Config

Base = declarative_base()

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), nullable=False)
    roll_number = Column(String(60), unique=True, nullable=False)
    embedding_json = Column(Text, nullable=False)  # JSON-encoded 512-d list of floats
    sample_count = Column(Integer, default=5)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    records = relationship("AttendanceRecord", back_populates="student", cascade="all, delete-orphan")

    def to_dict(self, anonymous: bool = False) -> Dict[str, Any]:
        display_name = f"Student #{self.id}" if anonymous else self.name
        display_roll = f"Roll #{self.id:03d}" if anonymous else self.roll_number
        return {
            "id": self.id,
            "name": display_name,
            "roll_number": display_roll,
            "sample_count": self.sample_count,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else ""
        }

class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(150), nullable=False)
    subject = Column(String(120), default="General")
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    duration_seconds = Column(Integer, default=0)
    avg_engagement = Column(Float, default=0.0)
    present_count = Column(Integer, default=0)
    total_enrolled = Column(Integer, default=0)

    attendance_records = relationship("AttendanceRecord", back_populates="session", cascade="all, delete-orphan")
    engagement_intervals = relationship("EngagementInterval", back_populates="session", cascade="all, delete-orphan")
    suggestions = relationship("TeachingSuggestion", back_populates="session", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "subject": self.subject,
            "started_at": self.started_at.strftime("%Y-%m-%d %H:%M:%S") if self.started_at else "",
            "ended_at": self.ended_at.strftime("%Y-%m-%d %H:%M:%S") if self.ended_at else "",
            "is_active": self.is_active,
            "duration_seconds": self.duration_seconds,
            "avg_engagement": round(self.avg_engagement, 1),
            "present_count": self.present_count,
            "total_enrolled": self.total_enrolled
        }

class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    first_seen = Column(DateTime, nullable=True)
    last_seen = Column(DateTime, nullable=True)
    total_presence_seconds = Column(Float, default=0.0)
    status = Column(String(30), default="Absent")  # "Present", "Late", "Partial", "Absent"
    sustained_verified = Column(Boolean, default=False)
    frames_present = Column(Integer, default=0)
    frames_total = Column(Integer, default=0)

    session = relationship("Session", back_populates="attendance_records")
    student = relationship("Student", back_populates="records")

    def to_dict(self, anonymous: bool = False, sustained_ratio: Optional[float] = None) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "student_id": self.student_id,
            "student_name": self.student.to_dict(anonymous)["name"] if self.student else "Unknown",
            "roll_number": self.student.to_dict(anonymous)["roll_number"] if self.student else "-",
            "first_seen": self.first_seen.strftime("%H:%M:%S") if self.first_seen else "--",
            "last_seen": self.last_seen.strftime("%H:%M:%S") if self.last_seen else "--",
            "total_presence_seconds": round(self.total_presence_seconds, 1),
            "status": self.status,
            "sustained_verified": self.sustained_verified,
            "sustained_ratio": round(sustained_ratio, 1) if sustained_ratio is not None else (100.0 if self.sustained_verified else 0.0)
        }

class EngagementInterval(Base):
    __tablename__ = "engagement_intervals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    interval_index = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    attention_score = Column(Float, default=0.0)
    movement_score = Column(Float, default=0.0)
    inactivity_score = Column(Float, default=0.0)
    audio_score = Column(Float, default=0.0)
    overall_engagement = Column(Float, default=0.0)
    speech_ratio = Column(Float, default=0.0)
    is_drop = Column(Boolean, default=False)
    is_peak = Column(Boolean, default=False)

    session = relationship("Session", back_populates="engagement_intervals")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interval_index": self.interval_index,
            "timestamp": self.timestamp.strftime("%H:%M:%S"),
            "attention": round(self.attention_score, 1),
            "movement": round(self.movement_score, 1),
            "inactivity": round(self.inactivity_score, 1),
            "audio": round(self.audio_score, 1),
            "engagement": round(self.overall_engagement, 1),
            "speech_ratio": round(self.speech_ratio, 2),
            "is_drop": self.is_drop,
            "is_peak": self.is_peak
        }

class TeachingSuggestion(Base):
    __tablename__ = "teaching_suggestions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    trigger_type = Column(String(50), default="drop")  # "drop", "summary", "periodic"
    drop_percentage = Column(Float, default=0.0)
    suggestion_text = Column(Text, nullable=False)
    source = Column(String(50), default="rule_engine")  # "ollama", "rule_engine"
    model_name = Column(String(50), default="RuleEngine")

    session = relationship("Session", back_populates="suggestions")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "time": self.timestamp.strftime("%H:%M:%S"),
            "trigger_type": self.trigger_type,
            "drop_percentage": round(self.drop_percentage, 1),
            "suggestion": self.suggestion_text,
            "source": self.source,
            "model_name": self.model_name
        }


class AttendanceManager:
    """
    Manages session lifecycle, real-time rolling window sustained presence,
    status transitions (Present, Late, Partial, Absent), and database persistence.
    """
    def __init__(self, db_uri: Optional[str] = None):
        uri = db_uri or Config.SQLALCHEMY_DATABASE_URI
        try:
            self.engine = create_engine(uri, echo=False, connect_args={"check_same_thread": False})
            Base.metadata.create_all(self.engine)
        except Exception as e:
            print(f"[AttendanceManager] Warning: failed to initialize SQLite at {uri} ({e}). Falling back to :memory: database.")
            self.engine = create_engine("sqlite:///:memory:", echo=False, connect_args={"check_same_thread": False})
            Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # In-memory tracking for active session rolling presence windows
        self.active_session_id: Optional[int] = None
        self.session_start_epoch: float = 0.0
        # Rolling window history per student: list of (epoch_time: float, was_detected: bool)
        self.student_window_history: Dict[int, List[tuple]] = {}
        # Cumulative presence time tracker: {student_id: float}
        self.student_presence_seconds: Dict[int, float] = {}
        self.last_frame_epoch: float = time.time()

        # Auto-activate live session so attendance recording is active immediately
        self._ensure_clean_startup_state()
        try:
            self.start_session("Live Classroom Lecture - CSE AIML", "AL-501")
            print("[AttendanceManager] Auto-started default session: Live Classroom Lecture (AL-501)")
        except Exception as ex:
            print(f"[AttendanceManager] Notice on auto-session: {ex}")

    def _ensure_clean_startup_state(self):
        """Cleanly archives any old dangling session so startup is strictly in Standby mode."""
        db = self.get_db()
        try:
            old_active = db.query(Session).filter(Session.is_active == True).all()
            if old_active:
                for s in old_active:
                    s.is_active = False
                    if not s.ended_at:
                        s.ended_at = datetime.utcnow()
                db.commit()
                print(f"[AttendanceManager] Archived {len(old_active)} dangling sessions. System ready in Standby mode.")
            self.active_session_id = None
            self.student_window_history = {}
            self.student_presence_seconds = {}
        except Exception as e:
            print(f"[AttendanceManager] Startup cleanup notice: {e}")
        finally:
            db.close()

    def get_db(self):
        db = self.SessionLocal()
        try:
            return db
        except Exception:
            db.close()
            raise

    def get_active_session(self) -> Optional[Dict[str, Any]]:
        db = self.get_db()
        try:
            active = db.query(Session).filter(Session.is_active == True).order_by(Session.id.desc()).first()
            return active.to_dict() if active else None
        finally:
            db.close()

    def start_session(self, name: str, subject: str) -> Dict[str, Any]:
        db = self.get_db()
        try:
            # End any running session
            running = db.query(Session).filter(Session.is_active == True).all()
            for r in running:
                r.is_active = False
                r.ended_at = datetime.utcnow()
            db.commit()

            # Count enrolled students
            all_students = db.query(Student).filter(Student.is_active == True).all()
            new_session = Session(
                name=name if name.strip() else f"Session {datetime.now().strftime('%b %d, %H:%M')}",
                subject=subject if subject.strip() else "Computer Science",
                started_at=datetime.utcnow(),
                is_active=True,
                total_enrolled=len(all_students)
            )
            db.add(new_session)
            db.commit()
            db.refresh(new_session)

            self.active_session_id = new_session.id
            self.session_start_epoch = time.time()
            self.student_window_history = {s.id: [] for s in all_students}
            self.student_presence_seconds = {s.id: 0.0 for s in all_students}
            self.last_frame_epoch = time.time()

            # Pre-populate AttendanceRecord for every student with status "Absent"
            for student in all_students:
                rec = AttendanceRecord(
                    session_id=new_session.id,
                    student_id=student.id,
                    status="Absent",
                    sustained_verified=False
                )
                db.add(rec)
            db.commit()

            return new_session.to_dict()
        finally:
            db.close()

    def process_frame(self, detected_student_ids: List[int], now_epoch: Optional[float] = None, anonymous: bool = False) -> List[Dict[str, Any]]:
        """
        Processes a single frame for sustained presence logic.
        Evaluates rolling presence ratio >= 0.70.
        Updates status to Present / Late / Partial.
        """
        now = now_epoch if now_epoch is not None else time.time()
        dt = max(0.01, min(1.0, now - self.last_frame_epoch))
        self.last_frame_epoch = now

        db = self.get_db()
        try:
            all_students = db.query(Student).filter(Student.is_active == True).all()
            if not all_students:
                return []

            # Standby / Live Preview when no formal session is active
            if self.active_session_id is None:
                preview_records = []
                for student in all_students:
                    sid = student.id
                    is_detected = (sid in detected_student_ids)
                    if is_detected:
                        self.student_presence_seconds[sid] = self.student_presence_seconds.get(sid, 0.0) + dt
                        now_str = datetime.utcnow().strftime("%H:%M:%S")
                    else:
                        now_str = "--"

                    sec = self.student_presence_seconds.get(sid, 0.0)
                    preview_records.append({
                        "id": sid,
                        "session_id": 0,
                        "student_id": sid,
                        "student_name": student.to_dict(anonymous)["name"],
                        "roll_number": student.to_dict(anonymous)["roll_number"],
                        "first_seen": now_str if is_detected else "--",
                        "last_seen": now_str if is_detected else "--",
                        "total_presence_seconds": round(sec, 1),
                        "status": "Ready (In Camera)" if is_detected else "Standby",
                        "sustained_verified": False,
                        "sustained_ratio": 0.0
                    })
                return preview_records

            # Active formal session tracking
            records = db.query(AttendanceRecord).filter(AttendanceRecord.session_id == self.active_session_id).all()
            rec_map = {r.student_id: r for r in records}
            window_cutoff = now - Config.ROLLING_WINDOW_SECONDS
            updated_records = []

            for student in all_students:
                sid = student.id
                if sid not in self.student_window_history:
                    self.student_window_history[sid] = []

                is_detected = (sid in detected_student_ids)
                self.student_window_history[sid].append((now, is_detected))

                # Prune history older than rolling window
                self.student_window_history[sid] = [
                    (t, det) for (t, det) in self.student_window_history[sid] if t >= window_cutoff
                ]

                # Overall window ratio
                window = self.student_window_history[sid]
                total_in_window = len(window)
                detected_in_window = sum(1 for (_, det) in window if det)
                overall_ratio = (detected_in_window / total_in_window) if total_in_window > 0 else 0.0

                # Recent 10-second active presence ratio
                recent_cutoff = now - 10.0
                recent_window = [det for (t, det) in window if t >= recent_cutoff]
                recent_ratio = (sum(1 for d in recent_window if d) / len(recent_window)) if recent_window else 0.0

                effective_ratio = max(overall_ratio, recent_ratio)

                rec = rec_map.get(sid)
                if not rec:
                    rec = AttendanceRecord(session_id=self.active_session_id, student_id=sid, status="Absent", sustained_verified=False)
                    db.add(rec)
                    db.flush()

                if is_detected:
                    self.student_presence_seconds[sid] = self.student_presence_seconds.get(sid, 0.0) + dt
                    rec.total_presence_seconds = self.student_presence_seconds[sid]
                    rec.last_seen = datetime.utcnow()
                    if rec.first_seen is None:
                        rec.first_seen = datetime.utcnow()

                rec.frames_total += 1
                if is_detected:
                    rec.frames_present += 1

                # Sustained presence trigger (verified if sustained over window or >= 5 seconds presence)
                if (len(recent_window) >= 8 and recent_ratio >= Config.SUSTAINED_PRESENCE_RATIO) or (total_in_window >= 5 and overall_ratio >= Config.SUSTAINED_PRESENCE_RATIO) or (rec.total_presence_seconds >= 5.0 and is_detected):
                    rec.sustained_verified = True
                    elapsed_since_start = now - self.session_start_epoch
                    if rec.status == "Absent" or rec.status == "Partial":
                        if elapsed_since_start <= Config.LATE_GRACE_PERIOD_SECONDS:
                            rec.status = "Present"
                        else:
                            rec.status = "Late"
                elif rec.sustained_verified:
                    pass
                elif is_detected or (rec.total_presence_seconds >= 1.5 and rec.status == "Absent"):
                    rec.status = "Partial"

                updated_records.append(rec.to_dict(anonymous=anonymous, sustained_ratio=effective_ratio * 100.0))

            db.commit()
            return updated_records
        finally:
            db.close()

    def end_session(self) -> Optional[Dict[str, Any]]:
        """
        Ends the current session, marks unseen enrolled students as Absent,
        calculates final present count, duration, and returns session summary.
        """
        if self.active_session_id is None:
            return None

        db = self.get_db()
        try:
            current_session = db.query(Session).filter(Session.id == self.active_session_id).first()
            if not current_session:
                return None

            now = datetime.utcnow()
            current_session.ended_at = now
            current_session.is_active = False

            if current_session.started_at:
                delta = (now - current_session.started_at).total_seconds()
                current_session.duration_seconds = int(delta)

            # Auto-mark any remaining unverified students as Absent
            records = db.query(AttendanceRecord).filter(AttendanceRecord.session_id == self.active_session_id).all()
            present_tally = 0
            for r in records:
                if r.status in ("Present", "Late"):
                    present_tally += 1
                elif not r.sustained_verified:
                    r.status = "Absent"

            current_session.present_count = present_tally

            # Average engagement from intervals
            intervals = db.query(EngagementInterval).filter(EngagementInterval.session_id == self.active_session_id).all()
            if intervals:
                avg_eng = sum(i.overall_engagement for i in intervals) / len(intervals)
                current_session.avg_engagement = round(avg_eng, 1)

            db.commit()
            result = current_session.to_dict()
            self.active_session_id = None
            return result
        finally:
            db.close()

    def get_live_attendance_table(self, session_id: Optional[int] = None, anonymous: bool = False) -> List[Dict[str, Any]]:
        target_id = session_id or self.active_session_id
        if not target_id:
            # Fallback to last session or list enrolled students
            db = self.get_db()
            try:
                last_sess = db.query(Session).order_by(Session.id.desc()).first()
                if last_sess:
                    target_id = last_sess.id
                else:
                    students = db.query(Student).filter(Student.is_active == True).all()
                    return [
                        {
                            "student_id": s.id,
                            "student_name": s.to_dict(anonymous)["name"],
                            "roll_number": s.to_dict(anonymous)["roll_number"],
                            "first_seen": "--",
                            "last_seen": "--",
                            "total_presence_seconds": 0.0,
                            "status": "Absent",
                            "sustained_verified": False
                        }
                        for s in students
                    ]
            finally:
                db.close()

        db = self.get_db()
        try:
            records = db.query(AttendanceRecord).filter(AttendanceRecord.session_id == target_id).all()
            return [r.to_dict(anonymous) for r in records]
        finally:
            db.close()
