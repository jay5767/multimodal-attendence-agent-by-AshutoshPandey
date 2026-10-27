"""Session management and asynchronous processing coordinator."""

import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import threading

from config.settings import DATA_DIR
from database.repository import repo
from core.video_processor import VideoProcessingPipeline
from utils.validators import validate_video_file
from utils.logging_config import logger


class SessionService:
    """Manages session creation, file storage, and execution of processing jobs."""

    def __init__(self):
        self._active_pipelines: Dict[int, VideoProcessingPipeline] = {}
        self._threads: Dict[int, threading.Thread] = {}

    def create_session(
        self,
        session_name: str,
        course_id: Optional[int],
        instructor: str,
        date: str,
        video_file_path: Optional[Path] = None
    ) -> int:
        """Create a new session record and prepare video file in data storage."""
        dest_video_path = None
        if video_file_path and video_file_path.exists():
            sessions_video_dir = DATA_DIR / "videos"
            sessions_video_dir.mkdir(parents=True, exist_ok=True)
            dest_video_path = sessions_video_dir / f"session_{video_file_path.name}"
            shutil.copy2(str(video_file_path), str(dest_video_path))

        session_id = repo.create_session(
            session_name=session_name,
            course_id=course_id,
            instructor=instructor,
            date=date,
            video_path=str(dest_video_path) if dest_video_path else None
        )
        logger.info(f"Created session {session_id}: '{session_name}'")
        return session_id

    def process_session_async(
        self,
        session_id: int,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        preview_callback: Optional[Callable] = None
    ) -> bool:
        """Launch video processing pipeline in background worker thread."""
        session = repo.get_session(session_id)
        if not session or not session.get("video_path"):
            logger.error(f"Cannot process session {session_id}: video path missing.")
            return False

        video_path = Path(session["video_path"])
        if not video_path.exists():
            logger.error(f"Video file does not exist: {video_path}")
            return False

        pipeline = VideoProcessingPipeline(
            progress_callback=progress_callback,
            frame_preview_callback=preview_callback
        )
        self._active_pipelines[session_id] = pipeline

        def worker():
            try:
                pipeline.process_session(session_id, video_path)
            except Exception as e:
                logger.error(f"Background worker failed for session {session_id}: {e}")
            finally:
                self._active_pipelines.pop(session_id, None)
                self._threads.pop(session_id, None)

        thread = threading.Thread(target=worker, daemon=True)
        self._threads[session_id] = thread
        thread.start()
        return True

    def process_session_sync(
        self,
        session_id: int,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        preview_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run video processing synchronously."""
        session = repo.get_session(session_id)
        if not session or not session.get("video_path"):
            raise ValueError("Session video path not found.")

        video_path = Path(session["video_path"])
        pipeline = VideoProcessingPipeline(
            progress_callback=progress_callback,
            frame_preview_callback=preview_callback
        )
        self._active_pipelines[session_id] = pipeline
        try:
            return pipeline.process_session(session_id, video_path)
        finally:
            self._active_pipelines.pop(session_id, None)

    def cancel_processing(self, session_id: int) -> bool:
        """Cancel an ongoing processing job."""
        pipeline = self._active_pipelines.get(session_id)
        if pipeline:
            pipeline.cancel()
            return True
        return False

    def is_processing(self, session_id: int) -> bool:
        return session_id in self._active_pipelines


session_service = SessionService()
