"""Timestamp and duration utility functions for time-windowed media analytics."""

from datetime import datetime, timezone
from typing import Union


def format_seconds_to_mmss(seconds: float) -> str:
    """Convert float seconds to MM:SS or HH:MM:SS format."""
    if seconds < 0:
        seconds = 0
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_current_iso_timestamp() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def calculate_window_index(timestamp_sec: float, window_size_sec: int) -> int:
    """Calculate the discrete integer index of a time window given seconds."""
    if window_size_sec <= 0:
        window_size_sec = 30
    return int(timestamp_sec // window_size_sec)


def get_window_bounds(window_index: int, window_size_sec: int) -> tuple[float, float]:
    """Return start and end seconds for a discrete window index."""
    start = float(window_index * window_size_sec)
    end = float((window_index + 1) * window_size_sec)
    return start, end
