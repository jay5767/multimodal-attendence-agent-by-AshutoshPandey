"""Initializes database with sample course, registered students, and a processed benchmark session."""

from pathlib import Path
from datetime import datetime
from database.repository import repo
from utils.sample_generator import generate_sample_classroom_video
from core.video_processor import VideoProcessingPipeline
from services.attendance_service import attendance_service
from services.report_service import report_service
from config.settings import SAMPLES_DIR
from utils.logging_config import logger


def seed_demo_environment():
    """Populate database with realistic initial classroom records and processed demo session."""
    logger.info("Seeding initial classroom intelligence data...")

    # 1. Create Demo Course
    course_id = repo.create_course(
        code="CS-302",
        name="Computer Vision & Classroom Intelligence",
        instructor="Prof. Elena Vance"
    )

    # 2. Register Consented Students
    students_data = [
        ("STU-2026-001", "Alice Vance", "alice.vance@university.edu"),
        ("STU-2026-002", "Brian Chen", "brian.chen@university.edu"),
        ("STU-2026-003", "Clara Morales", "clara.m@university.edu"),
        ("STU-2026-004", "David Okafor", "david.o@university.edu"),
    ]

    for sid, name, email in students_data:
        repo.create_student(
            student_id=sid,
            name=name,
            email=email,
            course_id=course_id,
            has_biometric_consent=True
        )

    # 3. Generate Benchmark Classroom Video
    demo_video = SAMPLES_DIR / "classroom_benchmark_lecture.mp4"
    if not demo_video.exists():
        generate_sample_classroom_video(demo_video, duration_seconds=15, fps=15)

    # 4. Create Session Record
    sess_id = repo.create_session(
        session_name="Lecture 1: Foundations of Multimodal Sensing",
        course_id=course_id,
        instructor="Prof. Elena Vance",
        date=datetime.now().strftime("%Y-%m-%d"),
        video_path=str(demo_video)
    )

    # 5. Process Session Pipeline
    pipeline = VideoProcessingPipeline()
    result = pipeline.process_session(sess_id, demo_video)

    # 6. Generate Initial PDF Report
    pdf_path = report_service.generate_session_pdf(sess_id)
    logger.info(f"Demo environment initialized successfully! Session #{sess_id} processed, PDF: {pdf_path.name}")
    return sess_id


if __name__ == "__main__":
    seed_demo_environment()
