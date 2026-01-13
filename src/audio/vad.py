"""
WebRTC VAD wrapper for real-time STT.

Purpose
-------
- Accepts fixed-size audio frames (recommended: 10/20/30 ms)
- Audio must be 16 kHz (or 8/16/32/48 kHz supported by WebRTC VAD)
- Returns a boolean speech/non-speech decision per frame
- Provides a simple "gate" helper to smooth decisions (start/end triggers)

Dependencies
------------
- webrtcvad-wheels (or webrtcvad)
- numpy

Important Notes
---------------
- WebRTC VAD expects 16-bit PCM bytes (little-endian).
- If your pipeline uses float32 audio in [-1, 1], convert to int16 first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import webrtcvad  # provided by webrtcvad-wheels or webrtcvad
except Exception as e:
    raise ImportError(
        "webrtcvad is not installed. Install 'webrtcvad-wheels' (Windows-friendly) or 'webrtcvad'."
    ) from e


SUPPORTED_SAMPLE_RATES = (8000, 16000, 32000, 48000)
SUPPORTED_FRAME_MS = (10, 20, 30)


def float32_to_int16_pcm(frame_f32: np.ndarray) -> bytes:
    """
    Convert float32 numpy array in [-1, 1] to 16-bit PCM bytes.

    - Clips to [-1, 1] to avoid overflow.
    - Returns little-endian int16 bytes.
    """
    if frame_f32.dtype != np.float32:
        frame_f32 = frame_f32.astype(np.float32, copy=False)

    x = np.clip(frame_f32, -1.0, 1.0)
    int16 = (x * 32767.0).astype(np.int16)
    return int16.tobytes()


@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    aggressiveness: int = 2  # 0 (least aggressive) .. 3 (most aggressive)
    # Smoothing / gating:
    speech_start_frames: int = 2      # how many consecutive speech frames to trigger "speech start"
    speech_end_silence_ms: int = 800  # how much continuous silence ends an utterance


class WebRTCVAD:
    """
    Thin wrapper around webrtcvad.Vad with convenience checks and conversion helpers.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        aggressiveness: int = 2,
    ):
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Unsupported sample_rate={sample_rate}. Use one of {SUPPORTED_SAMPLE_RATES}.")
        if frame_ms not in SUPPORTED_FRAME_MS:
            raise ValueError(f"Unsupported frame_ms={frame_ms}. Use one of {SUPPORTED_FRAME_MS}.")
        if not (0 <= aggressiveness <= 3):
            raise ValueError("aggressiveness must be in [0, 3].")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.aggressiveness = aggressiveness

        self._vad = webrtcvad.Vad(aggressiveness)
        self._samples_per_frame = int(sample_rate * frame_ms / 1000)

    @property
    def samples_per_frame(self) -> int:
        return self._samples_per_frame

    def is_speech(self, frame: np.ndarray) -> bool:
        """
        Determine if the given frame contains speech.

        Parameters
        ----------
        frame: np.ndarray
            Shape (samples,) float32 in [-1, 1] preferred.
            If provided as int16, we will treat it as PCM samples.

        Returns
        -------
        bool: True if speech, else False.
        """
        if frame.ndim != 1:
            # If caller passes (samples, channels), they should downmix before VAD
            raise ValueError("VAD expects a 1D mono frame. Downmix to mono before calling is_speech().")

        if frame.size != self._samples_per_frame:
            raise ValueError(
                f"Frame length mismatch: got {frame.size} samples, expected {self._samples_per_frame} "
                f"for {self.frame_ms}ms at {self.sample_rate}Hz."
            )

        # Convert to bytes PCM16
        if frame.dtype == np.int16:
            pcm16 = frame.tobytes()
        else:
            pcm16 = float32_to_int16_pcm(frame.astype(np.float32, copy=False))

        return bool(self._vad.is_speech(pcm16, self.sample_rate))


class VADGate:
    """
    A simple gate/endpointer built on top of per-frame VAD decisions.

    It provides:
    - in_speech state
    - speech_start trigger
    - speech_end trigger (based on silence duration)

    Typical use:
        gate = VADGate(...)
        for each frame:
            speech_start, speech_end, in_speech = gate.update(is_speech)
    """

    def __init__(
        self,
        frame_ms: int = 20,
        speech_start_frames: int = 2,
        speech_end_silence_ms: int = 800,
    ):
        if frame_ms not in SUPPORTED_FRAME_MS:
            raise ValueError(f"Unsupported frame_ms={frame_ms}. Use one of {SUPPORTED_FRAME_MS}.")
        if speech_start_frames < 1:
            raise ValueError("speech_start_frames must be >= 1.")
        if speech_end_silence_ms < frame_ms:
            raise ValueError("speech_end_silence_ms must be >= frame_ms.")

        self.frame_ms = frame_ms
        self.speech_start_frames = speech_start_frames
        self.speech_end_silence_ms = speech_end_silence_ms

        self.in_speech = False
        self._consec_speech = 0
        self._silence_ms = 0

    def reset(self) -> None:
        self.in_speech = False
        self._consec_speech = 0
        self._silence_ms = 0

    def update(self, is_speech: bool) -> Tuple[bool, bool, bool]:
        """
        Update gate with current frame's speech decision.

        Returns:
            (speech_start, speech_end, in_speech)
        """
        speech_start = False
        speech_end = False

        if is_speech:
            self._consec_speech += 1
            self._silence_ms = 0

            if not self.in_speech and self._consec_speech >= self.speech_start_frames:
                self.in_speech = True
                speech_start = True
        else:
            self._consec_speech = 0

            if self.in_speech:
                self._silence_ms += self.frame_ms
                if self._silence_ms >= self.speech_end_silence_ms:
                    self.in_speech = False
                    speech_end = True
                    # reset counters for next utterance
                    self._silence_ms = 0
                    self._consec_speech = 0

        return speech_start, speech_end, self.in_speech


# -----------------------------
# Quick standalone test
# -----------------------------
if __name__ == "__main__":
    # This quick test expects you to feed frames externally.
    # It only demonstrates the gate behavior.
    gate = VADGate(frame_ms=20, speech_start_frames=5, speech_end_silence_ms=800)
    decisions = [False]*10 + [True]*5 + [False]*50  # simulate a short utterance
    for i, d in enumerate(decisions):
        s0, s1, ins = gate.update(d)
        if s0:
            print(f"[{i}] SPEECH START")
        if s1:
            print(f"[{i}] SPEECH END")
