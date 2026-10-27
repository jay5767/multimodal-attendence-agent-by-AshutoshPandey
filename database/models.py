"""Database models, dataclasses, and schema definitions for SQLite."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


DATABASE_SCHEMA_SQL = """
-- Courses table
CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    instructor TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Students table
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    email TEXT,
    course_id INTEGER,
    face_encoding BLOB,
    face_encoding_json TEXT,
    has_biometric_consent INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(course_id) REFERENCES courses(id) ON DELETE SET NULL
);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_name TEXT NOT NULL,
    course_id INTEGER,
    instructor TEXT,
    date TEXT NOT NULL,
    duration_seconds REAL DEFAULT 0.0,
    video_path TEXT,
    audio_path TEXT,
    status TEXT DEFAULT 'created', -- 'created', 'processing', 'completed', 'failed'
    total_frames INTEGER DEFAULT 0,
    processed_frames INTEGER DEFAULT 0,
    fps REAL DEFAULT 0.0,
    resolution TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(course_id) REFERENCES courses(id) ON DELETE SET NULL
);

-- Attendance records table
CREATE TABLE IF NOT EXISTS attendance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    student_id TEXT, -- NULL for anonymous tracks
    track_id INTEGER NOT NULL,
    status TEXT NOT NULL, -- 'Present', 'Late', 'Absent', 'Needs Review', 'Manual'
    first_seen_timestamp REAL DEFAULT 0.0,
    last_seen_timestamp REAL DEFAULT 0.0,
    total_visible_seconds REAL DEFAULT 0.0,
    presence_ratio REAL DEFAULT 0.0,
    detection_count INTEGER DEFAULT 0,
    confidence_avg REAL DEFAULT 0.0,
    verification_type TEXT DEFAULT 'anonymous', -- 'face_matched', 'anonymous', 'manual_override'
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY(student_id) REFERENCES students(student_id) ON DELETE SET NULL
);

-- Visual measurements table (granular or sampled observations)
CREATE TABLE IF NOT EXISTS visual_measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    frame_idx INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_w INTEGER,
    bbox_h INTEGER,
    head_yaw REAL,
    head_pitch REAL,
    head_roll REAL,
    head_movement_speed REAL,
    body_movement_speed REAL,
    observable_state TEXT, -- 'forward_facing', 'turned_away', 'low_movement', 'active', 'unavailable'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Audio measurements table (windowed audio features)
CREATE TABLE IF NOT EXISTS audio_measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    is_speech INTEGER DEFAULT 0,
    speech_ratio REAL DEFAULT 0.0,
    energy_level REAL DEFAULT 0.0,
    silence_ratio REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Multimodal Fused session metrics (windowed composite indicators)
CREATE TABLE IF NOT EXISTS fused_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    window_start_sec REAL NOT NULL,
    window_end_sec REAL NOT NULL,
    visual_activity_index REAL DEFAULT 0.0,
    speech_activity_index REAL DEFAULT 0.0,
    composite_activity_score REAL DEFAULT 0.0,
    active_participants_count INTEGER DEFAULT 0,
    data_quality_index REAL DEFAULT 1.0,
    signal_availability TEXT, -- 'both', 'video_only', 'audio_only', 'none'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Pedagogical AI Recommendations table
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    timestamp_start_sec REAL NOT NULL,
    timestamp_end_sec REAL NOT NULL,
    category TEXT NOT NULL, -- 'Interaction Lull', 'Acoustic Surge', 'Movement Stillness', 'Pacing Opportunity'
    suggestion TEXT NOT NULL,
    evidence TEXT NOT NULL,
    confidence REAL DEFAULT 0.8,
    limitations TEXT,
    status TEXT DEFAULT 'pending', -- 'pending', 'accepted', 'dismissed'
    instructor_notes TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Processing logs
CREATE TABLE IF NOT EXISTS processing_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    log_level TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance_records(session_id);
CREATE INDEX IF NOT EXISTS idx_visual_session_time ON visual_measurements(session_id, timestamp_sec);
CREATE INDEX IF NOT EXISTS idx_audio_session_time ON audio_measurements(session_id, start_sec);
CREATE INDEX IF NOT EXISTS idx_fused_session_time ON fused_metrics(session_id, window_start_sec);
CREATE INDEX IF NOT EXISTS idx_recommendations_session ON recommendations(session_id);
"""


@dataclass
class Student:
    id: Optional[int] = None
    student_id: str = ""
    name: str = ""
    email: Optional[str] = None
    course_id: Optional[int] = None
    face_encoding_json: Optional[str] = None
    has_biometric_consent: bool = False
    is_active: bool = True
    created_at: Optional[str] = None


@dataclass
class Session:
    id: Optional[int] = None
    session_name: str = ""
    course_id: Optional[int] = None
    instructor: Optional[str] = None
    date: str = ""
    duration_seconds: float = 0.0
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    status: str = "created"
    total_frames: int = 0
    processed_frames: int = 0
    fps: float = 0.0
    resolution: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class AttendanceRecord:
    id: Optional[int] = None
    session_id: int = 0
    student_id: Optional[str] = None
    track_id: int = 0
    status: str = "Absent"
    first_seen_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0
    total_visible_seconds: float = 0.0
    presence_ratio: float = 0.0
    detection_count: int = 0
    confidence_avg: float = 0.0
    verification_type: str = "anonymous"
    notes: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class Recommendation:
    id: Optional[int] = None
    session_id: int = 0
    timestamp_start_sec: float = 0.0
    timestamp_end_sec: float = 0.0
    category: str = "General"
    suggestion: str = ""
    evidence: str = ""
    confidence: float = 0.85
    limitations: Optional[str] = None
    created_at: Optional[str] = None
