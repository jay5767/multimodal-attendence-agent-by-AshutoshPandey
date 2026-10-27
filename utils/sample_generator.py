"""Synthetic classroom video and audio generator for testing and demonstration.

Produces a standard MP4 video depicting classroom participants, instructional slides,
and an acoustic audio track so users can test the system without uploading private media.
"""

import cv2
import numpy as np
import wave
import struct
import math
from pathlib import Path
from typing import Optional
from config.settings import SAMPLES_DIR
from utils.logging_config import logger


def generate_sample_classroom_video(
    output_path: Optional[Path] = None,
    duration_seconds: int = 15,
    fps: int = 20,
    width: int = 640,
    height: int = 480
) -> Path:
    """Generate a realistic synthetic classroom lecture video with 3 student avatars."""
    if output_path is None:
        output_path = SAMPLES_DIR / "sample_classroom.mp4"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use MP4V codec compatible across platforms
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = duration_seconds * fps

    # Define 3 student positions at classroom desks
    students = [
        {"id": 1, "base_x": 140, "base_y": 280, "color": (190, 140, 90), "head_color": (210, 180, 140)},
        {"id": 2, "base_x": 320, "base_y": 290, "color": (80, 130, 200), "head_color": (220, 190, 150)},
        {"id": 3, "base_x": 500, "base_y": 275, "color": (120, 180, 100), "head_color": (205, 175, 135)}
    ]

    for f_idx in range(total_frames):
        # 1. Background classroom wall (soft warm slate)
        frame = np.full((height, width, 3), (235, 230, 225), dtype=np.uint8)

        # 2. Blackboard / Slide projection screen
        cv2.rectangle(frame, (80, 30), (width - 80, 180), (45, 52, 54), -1)
        cv2.rectangle(frame, (76, 26), (width - 76, 184), (180, 185, 190), 3)

        # Blackboard text
        cv2.putText(frame, "CS-101: Introduction to Multimodal AI", (100, 70), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "- Multi-Person Visual Tracking", (110, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 240, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "- Observable Head Orientation", (110, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 240, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "- Synchronized Audio Speech Activity", (110, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 240, 200), 1, cv2.LINE_AA)

        # 3. Classroom Desks
        cv2.rectangle(frame, (60, 350), (220, 460), (160, 120, 80), -1)
        cv2.rectangle(frame, (240, 350), (400, 460), (160, 120, 80), -1)
        cv2.rectangle(frame, (420, 350), (580, 460), (160, 120, 80), -1)

        # 4. Render student avatars with gentle natural movements
        t = f_idx / float(fps)
        for s in students:
            # Add small natural sway
            sway_x = int(math.sin(t * 1.5 + s["id"]) * 5)
            sway_y = int(math.cos(t * 1.0 + s["id"]) * 3)

            cx = s["base_x"] + sway_x
            cy = s["base_y"] + sway_y

            # Body / Torso
            cv2.ellipse(frame, (cx, cy + 50), (45, 60), 0, 0, 360, s["color"], -1)

            # Head
            head_cy = cy - 20
            cv2.circle(frame, (cx, head_cy), 26, s["head_color"], -1)

            # Hair
            cv2.ellipse(frame, (cx, head_cy - 10), (26, 16), 0, 180, 360, (50, 40, 35), -1)

            # Eyes (simulate looking at blackboard or taking notes)
            eye_offset_y = -3 if math.sin(t + s["id"]) > 0 else 3
            cv2.circle(frame, (cx - 8, head_cy + eye_offset_y), 3, (40, 40, 40), -1)
            cv2.circle(frame, (cx + 8, head_cy + eye_offset_y), 3, (40, 40, 40), -1)

            # Mouth
            cv2.line(frame, (cx - 5, head_cy + 12), (cx + 5, head_cy + 12), (70, 50, 50), 2)

        # 5. Timestamp overlay
        time_str = f"Time: {t:.1f}s / {duration_seconds}s"
        cv2.putText(frame, time_str, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    logger.info(f"Generated sample classroom video at {output_path}")

    # Generate companion audio WAV track
    wav_path = output_path.with_suffix(".wav")
    generate_sample_audio_track(wav_path, duration_seconds=duration_seconds)

    return output_path


def generate_sample_audio_track(output_path: Path, duration_seconds: int = 15, sample_rate: int = 16000) -> Path:
    """Generate a synthetic lecture audio WAV file with speech bursts and silence pauses."""
    total_samples = duration_seconds * sample_rate
    samples = np.zeros(total_samples, dtype=np.int16)

    # Alternate between speech segments and silence
    # Segment 0-4s: Lecture speech
    # Segment 4-7s: Silence (note-taking)
    # Segment 7-12s: Discussion speech
    # Segment 12-15s: Silence
    time_points = np.linspace(0, duration_seconds, total_samples, endpoint=False)

    for i, t in enumerate(time_points):
        in_speech = (0.5 <= t <= 4.5) or (7.0 <= t <= 12.0)
        if in_speech:
            # Multi-harmonic vocal simulation (150Hz pitch + harmonics)
            wave_val = (
                0.5 * math.sin(2 * math.pi * 180 * t) +
                0.3 * math.sin(2 * math.pi * 360 * t) +
                0.15 * math.sin(2 * math.pi * 540 * t)
            )
            # Add slight amplitude modulation
            wave_val *= (0.7 + 0.3 * math.sin(2 * math.pi * 3.5 * t))
            samples[i] = int(wave_val * 16000)
        else:
            # Low ambient classroom noise
            samples[i] = int(np.random.normal(0, 150))

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    logger.info(f"Generated sample companion audio track at {output_path}")
    return output_path
