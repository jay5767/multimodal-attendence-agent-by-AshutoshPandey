import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from config import Config
from modules.attendance.manager import AttendanceManager, Student

class EnrollmentManager:
    """
    Manages student enrollment, 5-sample quality feedback loop,
    embedding averaging and persistence, student editing and deletion,
    and privacy controls ('delete all data').
    """
    def __init__(self, attendance_manager: AttendanceManager):
        self.attendance_mgr = attendance_manager
        self.embeddings_path = Path(Config.EMBEDDINGS_FILE)
        # In-memory dictionary: {student_id: np.ndarray (normalized 512-d)}
        self.enrolled_embeddings: Dict[int, np.ndarray] = {}
        self.enrolled_students_cache: Dict[int, Dict[str, Any]] = {}
        self.reload_enrolled_embeddings()

    def reload_enrolled_embeddings(self):
        """Loads all active student embeddings from SQLite into memory for fast cosine search."""
        self.enrolled_embeddings.clear()
        self.enrolled_students_cache.clear()
        db = self.attendance_mgr.get_db()
        try:
            students = db.query(Student).filter(Student.is_active == True).all()
            if len(students) == 0:
                print("[EnrollmentManager] Database is fresh: auto-seeding classroom roster...")
                default_students = [
                    ("Ashutosh Pandey", "0103AL221030"),
                    ("Aarav Sharma", "0103AL221001"),
                    ("Priya Patel", "0103AL221045"),
                    ("Rohan Verma", "0103AL221055")
                ]
                for name, roll in default_students:
                    rng = np.random.RandomState(abs(hash(roll)) % (2**31))
                    dummy_vec = rng.randn(512).astype(np.float32)
                    dummy_vec /= np.linalg.norm(dummy_vec)
                    s_new = Student(
                        name=name,
                        roll_number=roll,
                        embedding_json=json.dumps(dummy_vec.tolist()),
                        sample_count=5,
                        is_active=True
                    )
                    db.add(s_new)
                db.commit()
                students = db.query(Student).filter(Student.is_active == True).all()

            for s in students:
                try:
                    vec = np.array(json.loads(s.embedding_json), dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        vec = vec / norm
                    self.enrolled_embeddings[s.id] = vec
                    self.enrolled_students_cache[s.id] = {
                        "id": s.id,
                        "name": s.name,
                        "roll_number": s.roll_number
                    }
                except Exception as ex:
                    print(f"[EnrollmentManager] Error parsing embedding for student {s.id}: {ex}")
        finally:
            db.close()

    def enroll_student(self, name: str, roll_number: str, sample_embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        Computes the average embedding across the 5 validated samples,
        normalizes it, persists to database, and refreshes the memory index.
        """
        if len(sample_embeddings) < Config.ENROLLMENT_SAMPLES_REQUIRED:
            raise ValueError(f"Requires exactly {Config.ENROLLMENT_SAMPLES_REQUIRED} face samples.")

        # Average and L2 normalize
        stacked = np.array(sample_embeddings, dtype=np.float32)
        mean_vec = np.mean(stacked, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-6:
            mean_vec = mean_vec / norm
        else:
            raise ValueError("Invalid face embedding vector.")

        db = self.attendance_mgr.get_db()
        try:
            # Check for existing student with same roll number
            existing = db.query(Student).filter(Student.roll_number == roll_number.strip()).first()
            if existing:
                existing.is_active = True
                existing.name = name.strip()
                existing.embedding_json = json.dumps(mean_vec.tolist())
                existing.sample_count = len(sample_embeddings)
                db.commit()
                student_id = existing.id
            else:
                new_student = Student(
                    name=name.strip(),
                    roll_number=roll_number.strip(),
                    embedding_json=json.dumps(mean_vec.tolist()),
                    sample_count=len(sample_embeddings),
                    is_active=True
                )
                db.add(new_student)
                db.commit()
                db.refresh(new_student)
                student_id = new_student.id

            self.reload_enrolled_embeddings()
            return {"id": student_id, "name": name, "roll_number": roll_number, "success": True}
        finally:
            db.close()

    def update_student(self, student_id: int, name: str, roll_number: str) -> bool:
        """Edits an existing student's name or roll number."""
        db = self.attendance_mgr.get_db()
        try:
            student = db.query(Student).filter(Student.id == student_id).first()
            if not student:
                return False
            student.name = name.strip()
            student.roll_number = roll_number.strip()
            db.commit()
            self.reload_enrolled_embeddings()
            return True
        finally:
            db.close()

    def delete_student(self, student_id: int) -> bool:
        """Deletes a student and their embeddings from the system."""
        db = self.attendance_mgr.get_db()
        try:
            student = db.query(Student).filter(Student.id == student_id).first()
            if not student:
                return False
            db.delete(student)
            db.commit()
            self.reload_enrolled_embeddings()
            return True
        finally:
            db.close()

    def delete_all_data(self) -> bool:
        """Privacy feature: permanently purges all students, sessions, embeddings, and records."""
        from modules.attendance.manager import Session, AttendanceRecord, EngagementInterval, TeachingSuggestion
        db = self.attendance_mgr.get_db()
        try:
            db.query(TeachingSuggestion).delete()
            db.query(EngagementInterval).delete()
            db.query(AttendanceRecord).delete()
            db.query(Session).delete()
            db.query(Student).delete()
            db.commit()
            self.reload_enrolled_embeddings()
            return True
        finally:
            db.close()

    def get_all_students(self, anonymous: bool = False) -> List[Dict[str, Any]]:
        """Returns list of enrolled students."""
        db = self.attendance_mgr.get_db()
        try:
            students = db.query(Student).filter(Student.is_active == True).order_by(Student.id.asc()).all()
            return [s.to_dict(anonymous) for s in students]
        finally:
            db.close()
