"""Audio activity analysis module.

Extracts classroom acoustic features, speech activity intervals, and silence ratios.
Does NOT identify individual speakers. Does NOT equate silence with disengagement.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import subprocess
import wave
import struct
import math
from config.settings import DEFAULT_CONFIG
from utils.logging_config import logger


@dataclass
class AudioIntervalMeasurement:
    """Classroom acoustic activity metrics for a discrete time window."""
    start_sec: float
    end_sec: float
    is_speech: bool
    speech_ratio: float
    energy_level: float
    silence_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "is_speech": 1 if self.is_speech else 0,
            "speech_ratio": round(self.speech_ratio, 3),
            "energy_level": round(self.energy_level, 4),
            "silence_ratio": round(self.silence_ratio, 3)
        }


class AudioAnalyzer:
    """Extracts and analyzes audio activity from recorded video files."""

    def __init__(self, energy_threshold: float = DEFAULT_CONFIG.engagement.energy_speech_threshold):
        self.energy_threshold = energy_threshold

    def extract_audio_track(self, video_path: Path, output_wav_path: Optional[Path] = None) -> Optional[Path]:
        """Extract audio track from video to uncompressed PCM 16-bit WAV using ffmpeg or fallback."""
        if not video_path.exists():
            return None

        if output_wav_path is None:
            output_wav_path = video_path.with_suffix(".wav")

        # If a companion or previously extracted WAV already exists
        if output_wav_path.exists() and output_wav_path.stat().st_size > 44:
            logger.info(f"Using companion audio track: {output_wav_path}")
            return output_wav_path

        # Locate ffmpeg executable
        import shutil
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            potential_paths = [
                Path(r"C:\Users\Ashutoshpandey\AppData\Local\Programs\LNV\Stremio-4\ffmpeg.exe"),
                Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
                Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
            ]
            for p in potential_paths:
                if p.exists():
                    ffmpeg_bin = str(p)
                    break

        if ffmpeg_bin:
            try:
                cmd = [
                    ffmpeg_bin, "-y", "-i", str(video_path),
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(output_wav_path)
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                if res.returncode == 0 and output_wav_path.exists() and output_wav_path.stat().st_size > 44:
                    logger.info(f"Audio extracted successfully via ffmpeg ({ffmpeg_bin}) to {output_wav_path}")
                    return output_wav_path
            except Exception as e:
                logger.warning(f"Error during ffmpeg audio extraction: {e}")

        # If video file has no audio or ffmpeg failed, return None to indicate no audio track
        logger.info(f"Video {video_path.name} has no separable audio track or ffmpeg not present.")
        return None

    def analyze_wav(
        self,
        wav_path: Path,
        total_duration_sec: float,
        window_size_sec: int = DEFAULT_CONFIG.engagement.window_size_seconds
    ) -> List[AudioIntervalMeasurement]:
        """Analyze a PCM WAV file across discrete time windows."""
        results = []
        if not wav_path.exists() or wav_path.stat().st_size <= 44:
            return self.generate_empty_windows(total_duration_sec, window_size_sec)

        try:
            with wave.open(str(wav_path), "rb") as wf:
                sample_rate = wf.getframerate()
                num_channels = wf.getnchannels()
                num_frames = wf.getnframes()
                sample_width = wf.getsampwidth()

                if sample_width != 2 or num_frames == 0:
                    return self.generate_empty_windows(total_duration_sec, window_size_sec)

                raw_data = wf.readframes(num_frames)
                total_samples = num_frames * num_channels
                # Unpack 16-bit signed integers
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                if num_channels > 1:
                    samples = samples[::num_channels]  # downsample to mono

                total_seconds = len(samples) / float(sample_rate)
                duration = max(total_duration_sec, total_seconds)
                num_windows = max(1, int(math.ceil(duration / window_size_sec)))

                samples_per_window = int(sample_rate * window_size_sec)

                for w_idx in range(num_windows):
                    start_sec = float(w_idx * window_size_sec)
                    end_sec = min(duration, float((w_idx + 1) * window_size_sec))
                    w_start_sample = int(w_idx * samples_per_window)
                    w_end_sample = min(len(samples), int((w_idx + 1) * samples_per_window))

                    chunk = samples[w_start_sample:w_end_sample]
                    if len(chunk) == 0:
                        results.append(AudioIntervalMeasurement(
                            start_sec=start_sec,
                            end_sec=end_sec,
                            is_speech=False,
                            speech_ratio=0.0,
                            energy_level=0.0,
                            silence_ratio=1.0
                        ))
                        continue

                    # Frame-level RMS energy within this window (e.g. 50ms sub-frames)
                    subframe_len = max(1, int(sample_rate * 0.05))
                    num_subframes = len(chunk) // subframe_len
                    if num_subframes == 0:
                        rms = float(np.sqrt(np.mean(chunk ** 2)))
                        is_speech = rms >= self.energy_threshold
                        speech_ratio = 1.0 if is_speech else 0.0
                    else:
                        subframes = chunk[:num_subframes * subframe_len].reshape(num_subframes, subframe_len)
                        sub_rms = np.sqrt(np.mean(subframes ** 2, axis=1))
                        active_subframes = np.sum(sub_rms >= self.energy_threshold)
                        speech_ratio = float(active_subframes / num_subframes)
                        rms = float(np.mean(sub_rms))

                    is_speech = speech_ratio >= 0.20
                    silence_ratio = max(0.0, 1.0 - speech_ratio)

                    results.append(AudioIntervalMeasurement(
                        start_sec=round(start_sec, 2),
                        end_sec=round(end_sec, 2),
                        is_speech=is_speech,
                        speech_ratio=round(speech_ratio, 3),
                        energy_level=round(rms, 4),
                        silence_ratio=round(silence_ratio, 3)
                    ))
        except Exception as e:
            logger.error(f"Error analyzing WAV file {wav_path}: {e}")
            return self.generate_empty_windows(total_duration_sec, window_size_sec)

        return results

    def generate_empty_windows(self, duration_sec: float, window_size_sec: int) -> List[AudioIntervalMeasurement]:
        """Generate empty interval measurements when audio is unavailable or silent."""
        duration = max(1.0, duration_sec)
        num_windows = max(1, int(math.ceil(duration / window_size_sec)))
        measurements = []
        for i in range(num_windows):
            s = float(i * window_size_sec)
            e = min(duration, float((i + 1) * window_size_sec))
            measurements.append(AudioIntervalMeasurement(
                start_sec=round(s, 2),
                end_sec=round(e, 2),
                is_speech=False,
                speech_ratio=0.0,
                energy_level=0.0,
                silence_ratio=1.0
            ))
        return measurements
