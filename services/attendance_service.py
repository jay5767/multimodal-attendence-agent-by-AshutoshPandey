"""Attendance business logic and student identity management service."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

from database.repository import repo
from core.face_recognition import FaceRecognizer
from utils.validators import validate_student_id
from utils.logging_config import logger


class AttendanceService:
    """Provides high-level business operations for attendance and student registration."""

    def __init__(self):
        self.face_recognizer = FaceRecognizer()

    def register_student(
        self,
        student_id: str,
        name: str,
        email: Optional[str] = None,
        course_id: Optional[int] = None,
        reference_image_bytes: Optional[bytes] = None,
        has_consent: bool = False
    ) -> Tuple[bool, str]:
        """Register a new student with optional consented reference face enrollment."""
        valid, err = validate_student_id(student_id)
        if not valid:
            return False, err or "Invalid student ID."

        if not name.strip():
            return False, "Student name is required."

        encoding_json = None

        if reference_image_bytes and has_consent:
            try:
                # Decode image bytes using OpenCV
                nparr = np.frombuffer(reference_image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    # Detect face in reference image
                    faces = self.face_recognizer.detect_faces(img)
                    if faces:
                        fx, fy, fw, fh = faces[0]
                        face_crop = img[fy:fy + fh, fx:fx + fw]
                        vec = self.face_recognizer.extract_face_feature_vector(face_crop)
                        if vec is not None:
                            encoding_json = json.dumps(vec.tolist())
                            logger.info(f"Enrolled biometric facial vector for student {student_id}")
                    else:
                        # Fallback to entire image if already cropped tightly
                        vec = self.face_recognizer.extract_face_feature_vector(img)
                        if vec is not None:
                            encoding_json = json.dumps(vec.tolist())
            except Exception as e:
                logger.error(f"Error processing reference image for student {student_id}: {e}")

        try:
            repo.create_student(
                student_id=student_id.strip(),
                name=name.strip(),
                email=email.strip() if email else None,
                course_id=course_id,
                face_encoding_json=encoding_json,
                has_biometric_consent=has_consent if encoding_json else False
            )
            return True, f"Student '{name}' registered successfully."
        except Exception as e:
            logger.error(f"Failed to register student: {e}")
            return False, f"Registration failed: {str(e)}"

    def get_student_statistics(self, student_id: str) -> Dict[str, Any]:
        """Compute attendance summary statistics for a student."""
        records = repo.get_student_attendance_history(student_id)
        total_sessions = len(records)
        if total_sessions == 0:
            return {
                "total_sessions": 0,
                "present_count": 0,
                "late_count": 0,
                "absent_count": 0,
                "attendance_rate": 0.0,
                "records": []
            }

        present_cnt = sum(1 for r in records if r["status"] == "Present")
        late_cnt = sum(1 for r in records if r["status"] == "Late")
        absent_cnt = sum(1 for r in records if r["status"] == "Absent")
        rate = round(((present_cnt + late_cnt) / total_sessions) * 100, 1)

        return {
            "total_sessions": total_sessions,
            "present_count": present_cnt,
            "late_count": late_cnt,
            "absent_count": absent_cnt,
            "attendance_rate": rate,
            "records": records
        }

    def manual_override(self, record_id: int, new_status: str, instructor_note: str) -> bool:
        """Apply manual instructor override to an attendance record."""
        try:
            repo.update_attendance_status(record_id, new_status, instructor_note)
            logger.info(f"Applied manual override to attendance record {record_id} -> {new_status}")
            return True
        except Exception as e:
            logger.error(f"Failed to apply override to record {record_id}: {e}")
            return False

    def purge_student_biometrics(self, student_id: str) -> bool:
        """Comply with biometric data deletion request."""
        return repo.purge_biometrics(student_id)


attendance_service = AttendanceService()
