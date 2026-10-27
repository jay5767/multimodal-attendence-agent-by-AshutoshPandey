"""Input and media validation helpers."""

from pathlib import Path
from typing import Tuple, Optional
from config.settings import DEFAULT_CONFIG


def validate_video_file(file_path: Path) -> Tuple[bool, Optional[str]]:
    """Validate that the provided path exists, has a valid extension, and is not empty."""
    if not file_path.exists():
        return False, f"File does not exist: {file_path.name}"

    ext = file_path.suffix.lower()
    if ext not in DEFAULT_CONFIG.video.supported_extensions:
        return False, f"Unsupported video format: '{ext}'. Allowed: {', '.join(DEFAULT_CONFIG.video.supported_extensions)}"

    size_bytes = file_path.stat().st_size
    if size_bytes == 0:
        return False, "File is empty (0 bytes)."

    max_bytes = DEFAULT_CONFIG.video.max_file_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        return False, f"File exceeds maximum size limit of {DEFAULT_CONFIG.video.max_file_size_mb} MB."

    return True, None


def validate_student_id(student_id: str) -> Tuple[bool, Optional[str]]:
    """Validate student ID format."""
    s = student_id.strip()
    if not s:
        return False, "Student ID cannot be empty."
    if len(s) < 2 or len(s) > 30:
        return False, "Student ID must be between 2 and 30 characters."
    return True, None
