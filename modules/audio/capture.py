import threading
import time
from typing import Dict, List, Optional, Any
import numpy as np
from config import Config
from modules.audio.vad import AudioVADAnalyzer

class AudioStreamManager:
    """
    Manages audio capture via host sounddevice (auto-detecting physical Realtek mic)
    and browser WebAudio API chunks.
    Provides real-time audio metrics for multimodal fusion.
    """
    def __init__(self, sample_rate: int = Config.AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.vad_analyzer = AudioVADAnalyzer(sample_rate=sample_rate, vad_mode=Config.AUDIO_VAD_MODE)
        self.is_recording = False
        self.stream = None
        self.buffer = []
        self.lock = threading.Lock()
        self.selected_device_name = "None"
        self.last_metrics = {
            "speech_ratio": 0.0,
            "rms_loudness": 0.0,
            "is_discussion_burst": False,
            "classification": "Silence / Quiet",
            "device_available": False,
            "device_name": "None",
            "source": "inactive"
        }

    def get_input_devices(self) -> List[Dict[str, Any]]:
        """Enumerates available microphone input devices."""
        devices = []
        try:
            import sounddevice as sd
            all_devs = sd.query_devices()
            for i, dev in enumerate(all_devs):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append({
                        "id": i,
                        "name": dev.get("name", f"Mic {i}"),
                        "channels": dev.get("max_input_channels"),
                        "sample_rate": dev.get("default_samplerate")
                    })
        except Exception as e:
            print(f"[AudioStreamManager] Could not enumerate audio devices: {e}")
        return devices

    def start_host_recording(self, device_id: Optional[int] = None) -> bool:
        """Starts background recording from physical local microphone."""
        try:
            import sounddevice as sd
            self.stop_recording()

            # Auto-detect physical hardware microphone if not specified
            if device_id is None:
                for i, dev in enumerate(sd.query_devices()):
                    if dev.get("max_input_channels", 0) > 0:
                        dname = dev.get("name", "").lower()
                        if "realtek" in dname or "array" in dname or "internal" in dname:
                            device_id = i
                            break
                        elif "iriun" not in dname and "voicemod" not in dname and "virtual" not in dname:
                            if device_id is None:
                                device_id = i

            dev_info = sd.query_devices()[device_id] if device_id is not None else {}
            self.selected_device_name = dev_info.get("name", "Default Microphone")

            def audio_callback(indata, frames, time_info, status):
                # Apply 6x soft gain multiplier for internal laptop microphone sensitivity
                boosted = np.clip(indata[:, 0].copy() * 6.0, -1.0, 1.0)
                with self.lock:
                    self.buffer.append(boosted)
                    if len(self.buffer) > 150:
                        self.buffer.pop(0)

            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=device_id,
                callback=audio_callback,
                blocksize=int(self.sample_rate * 0.03)  # 30ms block
            )
            self.stream.start()
            self.is_recording = True
            self.last_metrics["device_available"] = True
            self.last_metrics["device_name"] = self.selected_device_name
            self.last_metrics["source"] = "host_microphone"
            return True
        except Exception as e:
            print(f"[AudioStreamManager] Failed to start host recording: {e}")
            self.is_recording = False
            self.last_metrics["device_available"] = False
            return False

    def ingest_client_metric(self, rms: float, is_speech: bool):
        """Allows browser clients to push live WebAudio volume/speech metrics directly."""
        with self.lock:
            self.last_metrics["rms_loudness"] = round(float(rms), 4)
            if is_speech:
                current_ratio = self.last_metrics.get("speech_ratio", 0.0)
                self.last_metrics["speech_ratio"] = min(1.0, round(current_ratio * 0.8 + 0.25, 2))
                self.last_metrics["classification"] = "Discussion / Active Speech"
                self.last_metrics["is_discussion_burst"] = self.last_metrics["speech_ratio"] > 0.4
            else:
                current_ratio = self.last_metrics.get("speech_ratio", 0.0)
                self.last_metrics["speech_ratio"] = max(0.0, round(current_ratio * 0.85, 2))
                if self.last_metrics["speech_ratio"] < 0.08:
                    self.last_metrics["classification"] = "Silence / Quiet"
                    self.last_metrics["is_discussion_burst"] = False
            self.last_metrics["source"] = "browser_webaudio"

    def ingest_client_audio(self, pcm_data: np.ndarray):
        """Allows browser clients to push PCM audio samples directly over WebSocket."""
        with self.lock:
            self.buffer.append(pcm_data)
            if len(self.buffer) > 150:
                self.buffer.pop(0)
            self.last_metrics["source"] = "browser_webaudio"

    def stop_recording(self):
        """Stops the audio stream."""
        self.is_recording = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        with self.lock:
            self.buffer.clear()

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Processes buffered audio and returns latest audio metrics."""
        with self.lock:
            if not self.buffer:
                return {
                    "speech_ratio": self.last_metrics.get("speech_ratio", 0.0),
                    "rms_loudness": self.last_metrics.get("rms_loudness", 0.0),
                    "is_discussion_burst": self.last_metrics.get("is_discussion_burst", False),
                    "classification": self.last_metrics.get("classification", "Silence / Quiet"),
                    "source": self.last_metrics.get("source", "inactive")
                }
            concatenated = np.concatenate(self.buffer)
            retain_count = max(1, len(self.buffer) // 3)
            self.buffer = self.buffer[-retain_count:]

        metrics = self.vad_analyzer.process_buffer(concatenated)
        metrics["source"] = self.last_metrics.get("source", "active")
        self.last_metrics.update(metrics)
        return metrics
