"""Core video processing pipeline integrating detection, tracking, pose estimation, and audio analysis."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import time

from config.settings import DEFAULT_CONFIG
from core.person_detector import PersonDetector
from core.person_tracker import PersonTracker, TrackedPerson
from core.face_recognition import FaceRecognizer
from core.head_pose import HeadPoseEstimator
from core.audio_analyzer import AudioAnalyzer
from core.engagement_analyzer import EngagementAnalyzer
from core.multimodal_fusion import MultimodalFusionEngine
from core.recommendation_agent import RecommendationAgent
from database.repository import repo
from utils.validators import validate_video_file
from utils.logging_config import logger


class VideoProcessingPipeline:
    """End-to-end recorded classroom video processor."""

    def __init__(
        self,
        config=DEFAULT_CONFIG,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        frame_preview_callback: Optional[Callable[[np.ndarray, Dict[str, Any]], None]] = None
    ):
        self.config = config
        self.progress_callback = progress_callback
        self.frame_preview_callback = frame_preview_callback
        self.is_cancelled = False

        # Instantiate core ML/CV engines
        self.person_detector = PersonDetector(conf_threshold=config.detection.person_conf_threshold)
        self.person_tracker = PersonTracker(
            iou_threshold=config.detection.iou_tracker_threshold,
            max_lost_frames=config.detection.max_track_lost_frames,
            min_hits=config.detection.min_track_hits
        )
        self.face_recognizer = FaceRecognizer(similarity_threshold=config.attendance.face_similarity_threshold)
        self.head_pose_estimator = HeadPoseEstimator(config=config.engagement)
        self.audio_analyzer = AudioAnalyzer(energy_threshold=config.engagement.energy_speech_threshold)
        self.engagement_analyzer = EngagementAnalyzer(config=config.engagement)
        self.fusion_engine = MultimodalFusionEngine(fusion_config=config.fusion)
        self.recommendation_agent = RecommendationAgent()

    def cancel(self) -> None:
        """Signal the pipeline to stop gracefully."""
        self.is_cancelled = True
        logger.info("Video processing pipeline cancellation requested.")

    def process_session(self, session_id: int, video_path: Path) -> Dict[str, Any]:
        """Execute complete offline processing pipeline on a recorded classroom video."""
        self.is_cancelled = False
        valid, err = validate_video_file(video_path)
        if not valid:
            repo.update_session_status(session_id, "failed")
            raise ValueError(f"Video validation error: {err}")

        # Load enrolled consented students for face matching
        students = repo.list_students(active_only=True)
        self.face_recognizer.load_enrolled_students(students)

        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            repo.update_session_status(session_id, "failed")
            raise IOError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / video_fps if video_fps > 0 else 0.0

        repo.update_session_metadata(
            session_id=session_id,
            duration_seconds=duration_sec,
            fps=video_fps,
            total_frames=total_frames,
            resolution=f"{width}x{height}"
        )
        repo.update_session_status(session_id, "processing", 0, total_frames)

        logger.info(f"Processing session {session_id}: {total_frames} frames, {video_fps:.1f} FPS, {duration_sec:.1f}s duration.")

        # Determine frame skip step based on desired sample_rate_fps
        sample_fps = min(video_fps, max(0.5, self.config.video.sample_rate_fps))
        frame_step = max(1, int(round(video_fps / sample_fps)))

        visual_measurements_buffer: List[Dict[str, Any]] = []
        frame_idx = 0
        processed_count = 0
        latest_annotated_frame = None

        try:
            while cap.isOpened():
                if self.is_cancelled:
                    logger.warning("Pipeline cancelled by user.")
                    repo.update_session_status(session_id, "cancelled")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    timestamp_sec = frame_idx / video_fps

                    # 1. Person Detection
                    detections = self.person_detector.detect(frame)

                    # 2. Tracking
                    active_tracks = self.person_tracker.update(detections, timestamp_sec)

                    # 3. For each active tracked person, run face recognition & head pose
                    for track in active_tracks:
                        person_box = track.current_box

                        # Try face recognition if student not yet matched
                        if track.matched_student_id is None:
                            face_boxes = self.face_recognizer.detect_faces(frame, person_box)
                            if face_boxes:
                                fx, fy, fw, fh = face_boxes[0]
                                face_crop = frame[fy:fy + fh, fx:fx + fw]
                                match_res = self.face_recognizer.match_face(face_crop)
                                if match_res.is_matched:
                                    track.matched_student_id = match_res.student_id
                                    track.face_match_confidence = match_res.confidence

                        # Head pose & movement analysis
                        pose_res = self.head_pose_estimator.estimate(
                            frame=frame,
                            person_bbox=person_box,
                            track_id=track.track_id,
                            timestamp_sec=timestamp_sec
                        )

                        visual_measurements_buffer.append({
                            "session_id": session_id,
                            "timestamp_sec": round(timestamp_sec, 2),
                            "frame_idx": frame_idx,
                            "track_id": track.track_id,
                            "bbox_x": track.x1,
                            "bbox_y": track.y1,
                            "bbox_w": track.x2 - track.x1,
                            "bbox_h": track.y2 - track.y1,
                            "head_yaw": pose_res.yaw,
                            "head_pitch": pose_res.pitch,
                            "head_roll": pose_res.roll,
                            "head_movement_speed": pose_res.movement_speed,
                            "body_movement_speed": pose_res.movement_speed,
                            "observable_state": pose_res.observable_state
                        })

                    # Annotation for preview
                    latest_annotated_frame = self._annotate_frame(frame.copy(), active_tracks)
                    if self.frame_preview_callback and (processed_count % 3 == 0):
                        self.frame_preview_callback(latest_annotated_frame, {
                            "timestamp": timestamp_sec,
                            "active_count": len(active_tracks),
                            "frame": frame_idx
                        })

                    processed_count += 1
                    if self.progress_callback and total_frames > 0:
                        prog = min(0.70, (frame_idx / total_frames) * 0.70)
                        self.progress_callback(prog, f"Processing visual frames ({frame_idx}/{total_frames})...")

                frame_idx += 1

            cap.release()

            if self.is_cancelled:
                return {"status": "cancelled"}

            # Save visual measurements in batch
            if self.progress_callback:
                self.progress_callback(0.72, "Saving visual observations...")
            repo.save_visual_measurements_batch(visual_measurements_buffer)

            # Retrieve all persistent tracks across the session
            all_tracks = self.person_tracker.finalize()

            # 4. Generate Attendance Records
            if self.progress_callback:
                self.progress_callback(0.75, "Evaluating attendance thresholds...")
            attendance_records = self._calculate_attendance(session_id, all_tracks, duration_sec)
            repo.save_attendance_records(attendance_records)

            # 5. Audio Extraction & Analysis
            if self.progress_callback:
                self.progress_callback(0.80, "Analyzing audio signals...")
            audio_track_path = self.audio_analyzer.extract_audio_track(video_path)
            has_audio = audio_track_path is not None

            if has_audio:
                audio_measurements = self.audio_analyzer.analyze_wav(
                    audio_track_path,
                    total_duration_sec=duration_sec,
                    window_size_sec=self.config.engagement.window_size_seconds
                )
            else:
                audio_measurements = self.audio_analyzer.generate_empty_windows(
                    duration_sec=duration_sec,
                    window_size_sec=self.config.engagement.window_size_seconds
                )

            audio_records = []
            for m in audio_measurements:
                d = m.to_dict()
                d["session_id"] = session_id
                audio_records.append(d)
            repo.save_audio_measurements_batch(audio_records)

            # 6. Aggregate Visual Windows
            if self.progress_callback:
                self.progress_callback(0.85, "Aggregating engagement timelines...")
            visual_windows = self.engagement_analyzer.aggregate_visual_windows(
                measurements=visual_measurements_buffer,
                total_duration_sec=duration_sec,
                window_size_sec=self.config.engagement.window_size_seconds
            )

            # 7. Multimodal Fusion
            if self.progress_callback:
                self.progress_callback(0.90, "Fusing multimodal activity indicators...")
            fused_metrics = self.fusion_engine.fuse_signals(
                visual_windows=visual_windows,
                audio_windows=audio_measurements,
                total_duration_sec=duration_sec,
                has_audio_track=has_audio
            )
            fused_records = []
            for m in fused_metrics:
                d = m.to_dict()
                d["session_id"] = session_id
                fused_records.append(d)
            repo.save_fused_metrics_batch(fused_records)

            # 8. Pedagogical AI Teaching Recommendations
            if self.progress_callback:
                self.progress_callback(0.95, "Synthesizing AI teaching suggestions...")
            recommendations = self.recommendation_agent.generate_recommendations(
                fused_metrics=fused_metrics,
                session_duration_sec=duration_sec,
                has_audio=has_audio
            )
            rec_records = []
            for r in recommendations:
                d = r.to_dict()
                d["session_id"] = session_id
                rec_records.append(d)
            repo.save_recommendations_batch(rec_records)

            # Mark session as completed
            repo.update_session_status(session_id, "completed", processed_count, total_frames)
            if self.progress_callback:
                self.progress_callback(1.0, "Session analysis complete!")

            logger.info(f"Session {session_id} completed successfully. {len(attendance_records)} attendance tracks identified.")
            return {
                "status": "completed",
                "session_id": session_id,
                "duration_sec": duration_sec,
                "attendance_count": len(attendance_records),
                "fused_windows": len(fused_metrics),
                "recommendations_count": len(recommendations)
            }

        except Exception as e:
            logger.error(f"Error in video processing pipeline: {e}", exc_info=True)
            repo.update_session_status(session_id, "failed")
            raise

    def _calculate_attendance(
        self,
        session_id: int,
        tracks: List[TrackedPerson],
        duration_sec: float
    ) -> List[Dict[str, Any]]:
        """Calculate attendance records according to sustained-presence rules."""
        records = []
        eff_duration = max(1.0, duration_sec)
        required_min_seconds = min(self.config.attendance.min_visible_seconds, eff_duration * self.config.attendance.min_presence_ratio)

        for track in tracks:
            visible_sec = track.visible_duration_sec
            presence_ratio = visible_sec / eff_duration

            # Attendance status determination
            if presence_ratio >= self.config.attendance.min_presence_ratio and visible_sec >= required_min_seconds:
                # Check late arrival
                if track.first_seen_sec > (eff_duration * self.config.attendance.late_threshold_ratio):
                    status = "Late"
                else:
                    status = "Present"
            elif presence_ratio >= 0.20 or visible_sec >= (required_min_seconds * 0.4):
                status = "Needs Review"
            else:
                status = "Absent"

            verification_type = "face_matched" if track.matched_student_id else "anonymous"

            records.append({
                "session_id": session_id,
                "student_id": track.matched_student_id,
                "track_id": track.track_id,
                "status": status,
                "first_seen_timestamp": round(track.first_seen_sec, 2),
                "last_seen_timestamp": round(track.last_seen_sec, 2),
                "total_visible_seconds": round(visible_sec, 2),
                "presence_ratio": round(min(1.0, presence_ratio), 3),
                "detection_count": track.total_detections,
                "confidence_avg": round(track.confidence, 3),
                "verification_type": verification_type,
                "notes": f"Auto-computed (visible: {int(visible_sec)}s, ratio: {int(presence_ratio*100)}%)"
            })

        return records

    def _annotate_frame(self, frame: np.ndarray, tracks: List[TrackedPerson]) -> np.ndarray:
        """Draw bounding boxes and metadata overlays on video frame."""
        for track in tracks:
            x1, y1, x2, y2 = track.current_box
            sid = track.matched_student_id or f"Track #{track.track_id}"
            conf = int(track.confidence * 100)
            label = f"{sid} ({conf}%)"

            # Color scheme: Green for verified, Blue for anonymous
            color = (46, 204, 113) if track.matched_student_id else (241, 196, 15)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + len(label) * 9 + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return frame
