"""Data access repository for database transactions and queries."""

import json
from typing import List, Dict, Any, Optional
from database.db import db_manager
from database.models import Student, Session, AttendanceRecord, Recommendation
from utils.logging_config import logger


class Repository:
    """Provides strongly typed and transactional access to SQLite persistence."""

    def __init__(self, manager=db_manager):
        self.db = manager

    # ==========================
    # Courses
    # ==========================
    def create_course(self, code: str, name: str, instructor: str = "") -> int:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO courses (code, name, instructor) VALUES (?, ?, ?)",
                (code.strip(), name.strip(), instructor.strip())
            )
            conn.commit()
            if cursor.lastrowid:
                return cursor.lastrowid
            row = conn.execute("SELECT id FROM courses WHERE code = ?", (code.strip(),)).fetchone()
            return row["id"] if row else 1

    def list_courses(self) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute("SELECT * FROM courses ORDER BY name ASC").fetchall()
            return [dict(r) for r in rows]

    # ==========================
    # Students
    # ==========================
    def create_student(
        self,
        student_id: str,
        name: str,
        email: Optional[str] = None,
        course_id: Optional[int] = None,
        face_encoding_json: Optional[str] = None,
        has_biometric_consent: bool = False
    ) -> int:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO students (student_id, name, email, course_id, face_encoding_json, has_biometric_consent, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(student_id) DO UPDATE SET
                    name=excluded.name,
                    email=excluded.email,
                    course_id=excluded.course_id,
                    face_encoding_json=COALESCE(excluded.face_encoding_json, students.face_encoding_json),
                    has_biometric_consent=excluded.has_biometric_consent
                """,
                (student_id.strip(), name.strip(), email, course_id, face_encoding_json, 1 if has_biometric_consent else 0)
            )
            conn.commit()
            return cursor.lastrowid

    def list_students(self, course_id: Optional[int] = None, active_only: bool = True) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            query = "SELECT s.*, c.name as course_name FROM students s LEFT JOIN courses c ON s.course_id = c.id WHERE 1=1"
            params = []
            if active_only:
                query += " AND s.is_active = 1"
            if course_id:
                query += " AND s.course_id = ?"
                params.append(course_id)
            query += " ORDER BY s.name ASC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT s.*, c.name as course_name FROM students s LEFT JOIN courses c ON s.course_id = c.id WHERE s.student_id = ?",
                (student_id.strip(),)
            ).fetchone()
            return dict(row) if row else None

    def update_student(self, student_id: str, name: str, email: Optional[str], is_active: bool) -> None:
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE students SET name = ?, email = ?, is_active = ? WHERE student_id = ?",
                (name.strip(), email, 1 if is_active else 0, student_id.strip())
            )
            conn.commit()

    def purge_biometrics(self, student_id: str) -> bool:
        """Privacy compliance: completely purge enrolled facial encoding while keeping metadata."""
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE students SET face_encoding_json = NULL, face_encoding = NULL, has_biometric_consent = 0 WHERE student_id = ?",
                (student_id.strip(),)
            )
            conn.commit()
            logger.info(f"Purged biometric data for student {student_id}")
            return True

    def delete_student(self, student_id: str) -> bool:
        with self.db.get_connection() as conn:
            conn.execute("DELETE FROM students WHERE student_id = ?", (student_id.strip(),))
            conn.commit()
            return True

    # ==========================
    # Sessions
    # ==========================
    def create_session(
        self,
        session_name: str,
        course_id: Optional[int],
        instructor: str,
        date: str,
        video_path: Optional[str] = None,
        duration_seconds: float = 0.0,
        fps: float = 0.0,
        resolution: Optional[str] = None
    ) -> int:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sessions (session_name, course_id, instructor, date, video_path, duration_seconds, fps, resolution, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'created')
                """,
                (session_name.strip(), course_id, instructor.strip(), date, video_path, duration_seconds, fps, resolution)
            )
            conn.commit()
            return cursor.lastrowid

    def update_session_status(self, session_id: int, status: str, processed_frames: int = 0, total_frames: int = 0) -> None:
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET status = ?, processed_frames = ?, total_frames = ? WHERE id = ?",
                (status, processed_frames, total_frames, session_id)
            )
            conn.commit()

    def update_session_metadata(
        self,
        session_id: int,
        duration_seconds: float,
        fps: float,
        total_frames: int,
        resolution: str,
        audio_path: Optional[str] = None
    ) -> None:
        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE sessions SET duration_seconds = ?, fps = ?, total_frames = ?, resolution = ?, audio_path = ?
                WHERE id = ?
                """,
                (duration_seconds, fps, total_frames, resolution, audio_path, session_id)
            )
            conn.commit()

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT s.*, c.name as course_name, c.code as course_code FROM sessions s LEFT JOIN courses c ON s.course_id = c.id WHERE s.id = ?",
                (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_sessions(self, course_id: Optional[int] = None) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            query = "SELECT s.*, c.name as course_name FROM sessions s LEFT JOIN courses c ON s.course_id = c.id WHERE 1=1"
            params = []
            if course_id:
                query += " AND s.course_id = ?"
                params.append(course_id)
            query += " ORDER BY s.id DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def delete_session(self, session_id: int) -> bool:
        with self.db.get_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return True

    # ==========================
    # Attendance Records
    # ==========================
    def save_attendance_records(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        with self.db.get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO attendance_records (
                    session_id, student_id, track_id, status, first_seen_timestamp,
                    last_seen_timestamp, total_visible_seconds, presence_ratio,
                    detection_count, confidence_avg, verification_type, notes
                ) VALUES (
                    :session_id, :student_id, :track_id, :status, :first_seen_timestamp,
                    :last_seen_timestamp, :total_visible_seconds, :presence_ratio,
                    :detection_count, :confidence_avg, :verification_type, :notes
                )
                """,
                records
            )
            conn.commit()

    def get_attendance_for_session(self, session_id: int) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT a.*, s.name as student_name
                FROM attendance_records a
                LEFT JOIN students s ON a.student_id = s.student_id
                WHERE a.session_id = ?
                ORDER BY a.status ASC, a.track_id ASC
                """,
                (session_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def update_attendance_status(self, record_id: int, new_status: str, notes: str = "Manual instructor override") -> None:
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE attendance_records SET status = ?, verification_type = 'manual_override', notes = ? WHERE id = ?",
                (new_status, notes, record_id)
            )
            conn.commit()

    def get_student_attendance_history(self, student_id: str) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT a.*, s.session_name, s.date, s.duration_seconds as session_duration
                FROM attendance_records a
                JOIN sessions s ON a.session_id = s.id
                WHERE a.student_id = ?
                ORDER BY s.date DESC
                """,
                (student_id.strip(),)
            ).fetchall()
            return [dict(r) for r in rows]

    # ==========================
    # Visual & Audio Measurements
    # ==========================
    def save_visual_measurements_batch(self, measurements: List[Dict[str, Any]]) -> None:
        if not measurements:
            return
        with self.db.get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO visual_measurements (
                    session_id, timestamp_sec, frame_idx, track_id, bbox_x, bbox_y,
                    bbox_w, bbox_h, head_yaw, head_pitch, head_roll,
                    head_movement_speed, body_movement_speed, observable_state
                ) VALUES (
                    :session_id, :timestamp_sec, :frame_idx, :track_id, :bbox_x, :bbox_y,
                    :bbox_w, :bbox_h, :head_yaw, :head_pitch, :head_roll,
                    :head_movement_speed, :body_movement_speed, :observable_state
                )
                """,
                measurements
            )
            conn.commit()

    def get_visual_measurements_for_session(self, session_id: int) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM visual_measurements WHERE session_id = ? ORDER BY timestamp_sec ASC",
                (session_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_visual_measurements_for_window(self, session_id: int, start_sec: float, end_sec: float) -> List[Dict[str, Any]]:
        """Retrieve visual participant measurements within a specific time window."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM visual_measurements 
                WHERE session_id = ? AND timestamp_sec >= ? AND timestamp_sec < ?
                ORDER BY timestamp_sec ASC, track_id ASC
                """,
                (session_id, start_sec, end_sec)
            ).fetchall()
            return [dict(r) for r in rows]

    def save_audio_measurements_batch(self, measurements: List[Dict[str, Any]]) -> None:
        if not measurements:
            return
        with self.db.get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO audio_measurements (
                    session_id, start_sec, end_sec, is_speech, speech_ratio, energy_level, silence_ratio
                ) VALUES (
                    :session_id, :start_sec, :end_sec, :is_speech, :speech_ratio, :energy_level, :silence_ratio
                )
                """,
                measurements
            )
            conn.commit()

    def get_audio_measurements_for_session(self, session_id: int) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM audio_measurements WHERE session_id = ? ORDER BY start_sec ASC",
                (session_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ==========================
    # Fused Metrics & Recommendations
    # ==========================
    def save_fused_metrics_batch(self, metrics: List[Dict[str, Any]]) -> None:
        if not metrics:
            return
        with self.db.get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO fused_metrics (
                    session_id, window_start_sec, window_end_sec, visual_activity_index,
                    speech_activity_index, composite_activity_score, active_participants_count,
                    data_quality_index, signal_availability
                ) VALUES (
                    :session_id, :window_start_sec, :window_end_sec, :visual_activity_index,
                    :speech_activity_index, :composite_activity_score, :active_participants_count,
                    :data_quality_index, :signal_availability
                )
                """,
                metrics
            )
            conn.commit()

    def get_fused_metrics_for_session(self, session_id: int) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM fused_metrics WHERE session_id = ? ORDER BY window_start_sec ASC",
                (session_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def save_recommendations_batch(self, recommendations: List[Dict[str, Any]]) -> None:
        if not recommendations:
            return
        with self.db.get_connection() as conn:
            # Prepare records with default status/notes if omitted
            clean_recs = []
            for r in recommendations:
                r_copy = dict(r)
                if "status" not in r_copy or not r_copy["status"]:
                    r_copy["status"] = "pending"
                if "instructor_notes" not in r_copy:
                    r_copy["instructor_notes"] = ""
                clean_recs.append(r_copy)

            conn.executemany(
                """
                INSERT INTO recommendations (
                    session_id, timestamp_start_sec, timestamp_end_sec, category,
                    suggestion, evidence, confidence, limitations, status, instructor_notes
                ) VALUES (
                    :session_id, :timestamp_start_sec, :timestamp_end_sec, :category,
                    :suggestion, :evidence, :confidence, :limitations, :status, :instructor_notes
                )
                """,
                clean_recs
            )
            conn.commit()

    def get_recommendations_for_session(self, session_id: int, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            if status_filter:
                rows = conn.execute(
                    "SELECT * FROM recommendations WHERE session_id = ? AND status = ? ORDER BY timestamp_start_sec ASC",
                    (session_id, status_filter)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM recommendations WHERE session_id = ? ORDER BY timestamp_start_sec ASC",
                    (session_id,)
                ).fetchall()
            return [dict(r) for r in rows]

    def update_recommendation_status(self, rec_id: int, status: str, instructor_notes: Optional[str] = None) -> bool:
        """Update recommendation review state (e.g. accepted, dismissed) with optional notes."""
        with self.db.get_connection() as conn:
            if instructor_notes is not None:
                conn.execute(
                    "UPDATE recommendations SET status = ?, instructor_notes = ? WHERE id = ?",
                    (status, instructor_notes, rec_id)
                )
            else:
                conn.execute(
                    "UPDATE recommendations SET status = ? WHERE id = ?",
                    (status, rec_id)
                )
            conn.commit()
            return True

    def reassign_attendance_record(self, record_id: int, student_id: str, notes: Optional[str] = None) -> bool:
        """Reassign an anonymous track to a registered consented student."""
        with self.db.get_connection() as conn:
            notes_str = f"Reassigned to {student_id} by instructor. {notes or ''}".strip()
            conn.execute(
                """
                UPDATE attendance_records
                SET student_id = ?, verification_type = 'manual_override', notes = ?
                WHERE id = ?
                """,
                (student_id, notes_str, record_id)
            )
            conn.commit()
            return True

    # ==========================
    # Global Statistics
    # ==========================
    def get_dashboard_summary(self, session_id: Optional[int] = None) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            total_students = conn.execute("SELECT COUNT(*) as count FROM students WHERE is_active = 1").fetchone()["count"]
            total_sessions = conn.execute("SELECT COUNT(*) as count FROM sessions").fetchone()["count"]

            if not session_id:
                latest = conn.execute("SELECT id FROM sessions WHERE status = 'completed' ORDER BY id DESC LIMIT 1").fetchone()
                session_id = latest["id"] if latest else None

            session_data = None
            attendance_summary = {"present": 0, "absent": 0, "late": 0, "needs_review": 0, "total": 0, "attendance_pct": 0.0}
            avg_activity = 0.0

            if session_id:
                session_row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
                if session_row:
                    session_data = dict(session_row)

                att_rows = conn.execute("SELECT status, count(*) as count FROM attendance_records WHERE session_id = ? GROUP BY status", (session_id,)).fetchall()
                for r in att_rows:
                    st = r["status"].lower()
                    cnt = r["count"]
                    if "present" in st:
                        attendance_summary["present"] += cnt
                    elif "absent" in st:
                        attendance_summary["absent"] += cnt
                    elif "late" in st:
                        attendance_summary["late"] += cnt
                    else:
                        attendance_summary["needs_review"] += cnt
                    attendance_summary["total"] += cnt

                if attendance_summary["total"] > 0:
                    attendance_summary["attendance_pct"] = round(
                        (attendance_summary["present"] + attendance_summary["late"]) / attendance_summary["total"] * 100, 1
                    )

                fused_avg = conn.execute(
                    "SELECT AVG(composite_activity_score) as avg_score FROM fused_metrics WHERE session_id = ?",
                    (session_id,)
                ).fetchone()
                if fused_avg and fused_avg["avg_score"] is not None:
                    avg_activity = round(float(fused_avg["avg_score"]), 2)

            return {
                "total_students": total_students,
                "total_sessions": total_sessions,
                "selected_session": session_data,
                "attendance_summary": attendance_summary,
                "avg_activity_index": avg_activity
            }


repo = Repository()
