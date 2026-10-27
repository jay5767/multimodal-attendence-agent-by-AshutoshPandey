import numpy as np
from typing import Dict, Any, List
from config import Config

class AudioVADAnalyzer:
    """
    Analyzes raw audio PCM buffers for speech activity, loudness (RMS),
    speech ratio over time windows, and interaction bursts.
    """
    def __init__(self, sample_rate: int = 16000, vad_mode: int = 2):
        self.sample_rate = sample_rate
        self.vad_mode = vad_mode
        self.vad = None
        self.use_vad = False

        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(vad_mode)
            self.use_vad = True
        except Exception as e:
            print(f"[AudioVAD] WebRTC VAD fallback to energy threshold: {e}")
            self.use_vad = False

        self.recent_speech_flags: List[bool] = []
        self.recent_rms_values: List[float] = []

    def compute_rms(self, samples: np.ndarray) -> float:
        """Calculates Root Mean Square (RMS) loudness."""
        if len(samples) == 0:
            return 0.0
        # If integer PCM, normalize to [-1, 1]
        if samples.dtype in (np.int16, np.int32):
            samples = samples.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(samples))))
        return rms

    def is_speech_chunk(self, chunk_pcm16: bytes) -> bool:
        """Evaluates whether a 10/20/30ms PCM16 buffer contains human speech."""
        if self.use_vad and self.vad is not None:
            try:
                return self.vad.is_speech(chunk_pcm16, self.sample_rate)
            except Exception:
                pass
        
        # Energy-based fallback
        samples = np.frombuffer(chunk_pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(samples))))
        return rms > Config.AUDIO_RMS_THRESHOLD

    def process_buffer(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Processes a block of audio (typically 1 to 5 seconds of 16kHz audio).
        Returns speech ratio, average RMS loudness, and interaction classification.
        """
        if len(audio_data) == 0:
            return {
                "speech_ratio": 0.0,
                "rms_loudness": 0.0,
                "is_discussion_burst": False,
                "classification": "Silence"
            }

        # Ensure float32 normalized and int16 pcm
        if audio_data.dtype != np.int16:
            # Assume float in [-1, 1]
            samples_float = audio_data.astype(np.float32)
            samples_int16 = (np.clip(samples_float, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            samples_int16 = audio_data
            samples_float = audio_data.astype(np.float32) / 32768.0

        rms = self.compute_rms(samples_float)
        self.recent_rms_values.append(rms)
        if len(self.recent_rms_values) > 30:
            self.recent_rms_values.pop(0)

        # Split into 30ms frames for VAD (16000 * 0.03 = 480 samples = 960 bytes)
        frame_size = int(self.sample_rate * 0.03)
        num_frames = len(samples_int16) // frame_size
        speech_count = 0

        for i in range(num_frames):
            frame = samples_int16[i * frame_size : (i + 1) * frame_size].tobytes()
            is_speech = self.is_speech_chunk(frame)
            if is_speech:
                speech_count += 1
            self.recent_speech_flags.append(is_speech)
            if len(self.recent_speech_flags) > 100:
                self.recent_speech_flags.pop(0)

        speech_ratio = float(speech_count / max(1, num_frames))

        # Classification
        classification = "Silence"
        is_burst = False
        if rms < Config.AUDIO_RMS_THRESHOLD or speech_ratio < 0.10:
            classification = "Silence / Quiet"
        elif speech_ratio >= Config.DISCUSSION_BURST_THRESHOLD:
            classification = "Discussion Burst"
            is_burst = True
        else:
            classification = "Teacher Talk / Lecture"

        return {
            "speech_ratio": round(speech_ratio, 3),
            "rms_loudness": round(rms, 4),
            "is_discussion_burst": is_burst,
            "classification": classification
        }
