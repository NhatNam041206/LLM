"""
Microphone input module for real-time STT.

- Captures audio from the default (or selected) input device
- Produces frames as numpy float32 arrays
- Designed for streaming pipelines (VAD/STT) that want fixed frame sizes

Primary backend: sounddevice (recommended)
Fallback backend: PyAudio (optional)

Usage (sounddevice):
    from audio.mic_input import MicInput

    mic = MicInput(sample_rate=16000, channels=1, frame_ms=20)
    mic.start()
    try:
        for frame in mic.frames():
            # frame: np.ndarray shape (n_samples,) if channels=1 else (n_samples, channels)
            ...
    finally:
        mic.stop()
"""

from __future__ import annotations

import time
import queue
from dataclasses import dataclass
from typing import Generator, Optional, List, Dict, Any

import numpy as np


@dataclass
class MicConfig:
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 20
    device: Optional[int] = None  # sounddevice device index or PyAudio device index
    backend: str = "sounddevice"   # "sounddevice" or "pyaudio"
    dtype: str = "float32"         # "float32" recommended
    queue_maxsize: int = 100       # prevents unbounded memory growth


class MicInput:
    def __init__(self, sample_rate: int = 16000, channels: int = 1, frame_ms: int = 20,
                 device: Optional[int] = None, backend: str = "sounddevice",
                 dtype: str = "float32", queue_maxsize: int = 100):
        self.cfg = MicConfig(
            sample_rate=sample_rate,
            channels=channels,
            frame_ms=frame_ms,
            device=device,
            backend=backend.lower(),
            dtype=dtype,
            queue_maxsize=queue_maxsize,
        )
        self._frames_per_buffer = int(self.cfg.sample_rate * self.cfg.frame_ms / 1000)
        if self._frames_per_buffer <= 0:
            raise ValueError("frame_ms too small; frames_per_buffer computed <= 0")

        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=self.cfg.queue_maxsize)
        self._running = False

        # sounddevice state
        self._sd_stream = None

        # pyaudio state
        self._pa = None
        self._pa_stream = None

    # -----------------------------
    # Public helpers
    # -----------------------------
    @staticmethod
    def list_input_devices_sounddevice() -> List[Dict[str, Any]]:
        """List input devices using sounddevice."""
        import sounddevice as sd
        devices = sd.query_devices()
        out = []
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                out.append({"index": i, "name": d.get("name"), "max_input_channels": d.get("max_input_channels")})
        return out

    @staticmethod
    def list_input_devices_pyaudio() -> List[Dict[str, Any]]:
        """List input devices using PyAudio."""
        import pyaudio
        pa = pyaudio.PyAudio()
        out = []
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0:
                    out.append({
                        "index": i,
                        "name": info.get("name"),
                        "max_input_channels": int(info.get("maxInputChannels", 0)),
                        "default_sample_rate": int(info.get("defaultSampleRate", 0)),
                    })
        finally:
            pa.terminate()
        return out

    def frames(self, timeout_s: float = 1.0) -> Generator[np.ndarray, None, None]:
        """
        Generator yielding audio frames as numpy arrays.

        - If channels == 1: returns shape (n_samples,)
        - Else: returns shape (n_samples, channels)

        timeout_s controls how long to wait for a frame before checking running state again.
        """
        while self._running:
            try:
                frame = self._q.get(timeout=timeout_s)
            except queue.Empty:
                continue
            yield frame

    def start(self) -> None:
        """Start microphone streaming."""
        if self._running:
            return
        self._running = True

        if self.cfg.backend == "sounddevice":
            self._start_sounddevice()
        elif self.cfg.backend == "pyaudio":
            self._start_pyaudio()
        else:
            self._running = False
            raise ValueError(f"Unknown backend '{self.cfg.backend}'. Use 'sounddevice' or 'pyaudio'.")

    def stop(self) -> None:
        """Stop microphone streaming and release resources."""
        self._running = False

        # Drain queue to free memory
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
            except Exception:
                pass
            try:
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None

        if self._pa_stream is not None:
            try:
                self._pa_stream.stop_stream()
            except Exception:
                pass
            try:
                self._pa_stream.close()
            except Exception:
                pass
            self._pa_stream = None

        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def __enter__(self) -> "MicInput":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # -----------------------------
    # sounddevice backend
    # -----------------------------
    def _start_sounddevice(self) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            if not self._running:
                return
            if status:
                # Status can indicate over/under-runs; not fatal but useful for debugging
                # You can log this elsewhere if needed.
                pass

            # indata: shape (frames, channels) float32 (if dtype float32)
            arr = np.asarray(indata, dtype=np.float32)

            if self.cfg.channels == 1:
                arr = arr.reshape(-1)  # (frames,)

            # Push to queue without blocking; if full, drop the oldest (keep most recent)
            try:
                self._q.put_nowait(arr)
            except queue.Full:
                try:
                    _ = self._q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(arr)
                except queue.Full:
                    pass  # if still full, drop

        self._sd_stream = sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype=self.cfg.dtype,
            device=self.cfg.device,
            blocksize=self._frames_per_buffer,  # ensures frame_ms-sized chunks
            callback=callback,
        )
        self._sd_stream.start()

    # -----------------------------
    # PyAudio backend
    # -----------------------------
    def _start_pyaudio(self) -> None:
        import pyaudio

        # Note: PyAudio callback receives bytes; we decode into float32
        # We'll open as float32 if supported; otherwise int16 and convert.
        self._pa = pyaudio.PyAudio()

        # Prefer float32 stream; fall back to int16 if needed
        pa_format = pyaudio.paFloat32
        bytes_per_sample = 4

        def _open_stream(fmt):
            return self._pa.open(
                format=fmt,
                channels=self.cfg.channels,
                rate=self.cfg.sample_rate,
                input=True,
                input_device_index=self.cfg.device,
                frames_per_buffer=self._frames_per_buffer,
                stream_callback=self._pyaudio_callback_factory(fmt),
            )

        try:
            self._pa_stream = _open_stream(pa_format)
        except Exception:
            pa_format = pyaudio.paInt16
            bytes_per_sample = 2
            self._pa_stream = _open_stream(pa_format)

        self._pa_stream.start_stream()

        # Keep local vars for clarity (not strictly required)
        _ = bytes_per_sample

    def _pyaudio_callback_factory(self, pa_format):
        import pyaudio

        def callback(in_data, frame_count, time_info, status_flags):
            if not self._running:
                return (None, pyaudio.paComplete)

            if pa_format == pyaudio.paFloat32:
                arr = np.frombuffer(in_data, dtype=np.float32)
            else:
                # int16 -> float32 in [-1, 1]
                int16 = np.frombuffer(in_data, dtype=np.int16)
                arr = (int16.astype(np.float32) / 32768.0)

            if self.cfg.channels > 1:
                arr = arr.reshape(-1, self.cfg.channels)
            else:
                arr = arr.reshape(-1)

            # Queue push with drop-oldest strategy
            try:
                self._q.put_nowait(arr)
            except queue.Full:
                try:
                    _ = self._q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(arr)
                except queue.Full:
                    pass

            return (None, pyaudio.paContinue)

        return callback


# ---------------------------------------
# Quick standalone test (optional)
# ---------------------------------------
if __name__ == "__main__":
    # Minimal sanity test: print RMS level
    mic = MicInput(sample_rate=16000, channels=1, frame_ms=20, backend="sounddevice")
    mic.start()
    print("Listening... (Ctrl+C to stop)")
    try:
        for frame in mic.frames():
            rms = float(np.sqrt(np.mean(frame ** 2))) if frame.size else 0.0
            print(f"\rRMS: {rms:.4f}", end="", flush=True)
            time.sleep(0.0)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        print("\nStopped.")
