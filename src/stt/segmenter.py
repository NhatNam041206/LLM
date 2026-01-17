"""
Speech segmenter (endpointer) built on top of VAD decisions.

Purpose
-------
Given per-frame VAD outputs (speech / silence), this module:
- Detects utterance start
- Detects utterance end after sustained silence (hangover)
- Optionally provides an audio buffer for the current utterance

This is meant to be used in a streaming STT pipeline, typically like:

    seg = SpeechSegmenter(sample_rate=16000, frame_ms=20, start_trigger_frames=2, end_silence_ms=800)

    for frame in mic_frames:
        is_speech = vad.is_speech(frame)
        event = seg.update(frame, is_speech)

        if event.type == SegmentEventType.SPEECH_START:
            ...
        elif event.type == SegmentEventType.SPEECH_END:
            audio = event.audio  # utterance audio (float32 mono)
            ...

Notes
-----
- This segmenter is deliberately simple, reliable, and low-memory.
- For "Option B" rolling-window streaming you may not need to store utterance audio,
  but it is still useful for:
    * utterance-mode STT (transcribe once at end)
    * producing a final utterance audio slice for post-processing
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List

import numpy as np


class SegmentEventType(Enum):
    NONE = auto()
    SPEECH_START = auto()
    SPEECH_END = auto()


@dataclass
class SegmentEvent:
    type: SegmentEventType
    # Monotonic time in ms since segmenter start (approx based on frames)
    t_ms: int
    # The audio for the utterance (only present on SPEECH_END if store_utterance_audio=True)
    audio: Optional[np.ndarray] = None
    # Optional info
    frames: int = 0  # frames in utterance


@dataclass
class SegmenterConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    start_trigger_frames: int = 2   # consecutive speech frames to trigger start
    end_silence_ms: int = 800       # silence hangover to trigger end
    min_utterance_ms: int = 200     # discard very short utterances (noise/clicks)
    max_utterance_ms: int = 10_000  # hard cap to prevent overly long speeches (10 seconds)
    store_utterance_audio: bool = True


class SpeechSegmenter:
    """
    Stateful speech segmenter. Call update() once per frame.

    Inputs:
        frame: np.ndarray float32 mono, shape (samples,)
        is_speech: bool

    Outputs:
        SegmentEvent (NONE / SPEECH_START / SPEECH_END)
    """

    def __init__(self, cfg: SegmenterConfig):
        if cfg.frame_ms <= 0:
            raise ValueError("frame_ms must be > 0")
        if cfg.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if cfg.start_trigger_frames < 1:
            raise ValueError("start_trigger_frames must be >= 1")
        if cfg.end_silence_ms < cfg.frame_ms:
            raise ValueError("end_silence_ms must be >= frame_ms")

        self.cfg = cfg
        self.samples_per_frame = int(cfg.sample_rate * cfg.frame_ms / 1000)

        # internal state
        self._t_ms = 0
        self._in_speech = False
        self._consec_speech = 0
        self._silence_ms = 0

        self._utt_frames = 0
        self._utt_audio_chunks: List[np.ndarray] = []

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def t_ms(self) -> int:
        return self._t_ms

    def reset(self) -> None:
        self._t_ms = 0
        self._in_speech = False
        self._consec_speech = 0
        self._silence_ms = 0
        self._utt_frames = 0
        self._utt_audio_chunks.clear()

    def _start_utterance(self) -> SegmentEvent:
        self._in_speech = True
        self._silence_ms = 0
        self._utt_frames = 0
        self._utt_audio_chunks.clear()
        return SegmentEvent(type=SegmentEventType.SPEECH_START, t_ms=self._t_ms)

    def _end_utterance(self) -> SegmentEvent:
        # Build audio if we stored it
        audio = None
        if self.cfg.store_utterance_audio and self._utt_audio_chunks:
            audio = np.concatenate(self._utt_audio_chunks, axis=0)

        utt_ms = self._utt_frames * self.cfg.frame_ms

        # Reset to ready state
        self._in_speech = False
        self._consec_speech = 0
        self._silence_ms = 0

        event = SegmentEvent(
            type=SegmentEventType.SPEECH_END,
            t_ms=self._t_ms,
            audio=audio,
            frames=self._utt_frames,
        )

        # Clear utterance buffers
        self._utt_frames = 0
        self._utt_audio_chunks.clear()

        # Filter too-short utterances (return NONE instead of SPEECH_END)
        if utt_ms < self.cfg.min_utterance_ms:
            return SegmentEvent(type=SegmentEventType.NONE, t_ms=self._t_ms)

        return event

    def update(self, frame: np.ndarray, is_speech: bool) -> SegmentEvent:
        """
        Update the segmenter with one audio frame and its VAD decision.

        frame:
            np.ndarray float32 mono, shape (samples,)
        is_speech:
            bool from VAD

        Returns:
            SegmentEvent
        """
        # time advances per frame
        self._t_ms += self.cfg.frame_ms

        # Validate frame shape
        if frame.ndim != 1:
            raise ValueError("SpeechSegmenter expects mono 1D frames.")
        if frame.size != self.samples_per_frame:
            raise ValueError(
                f"Frame size mismatch: got {frame.size}, expected {self.samples_per_frame} "
                f"for {self.cfg.frame_ms}ms at {self.cfg.sample_rate}Hz."
            )
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32, copy=False)

        # If not currently in an utterance, look for start trigger
        if not self._in_speech:
            if is_speech:
                self._consec_speech += 1
                if self._consec_speech >= self.cfg.start_trigger_frames:
                    # start utterance; include this frame (and optionally previous ones if you keep a pre-roll)
                    event = self._start_utterance()
                    # also record this frame as first utterance audio
                    if self.cfg.store_utterance_audio:
                        self._utt_audio_chunks.append(frame.copy())
                    self._utt_frames += 1
                    return event
            else:
                self._consec_speech = 0

            return SegmentEvent(type=SegmentEventType.NONE, t_ms=self._t_ms)

        # We are in an utterance
        # Store audio (always store, speech or silence, until we end)
        if self.cfg.store_utterance_audio:
            self._utt_audio_chunks.append(frame.copy())
        self._utt_frames += 1

        utt_ms = self._utt_frames * self.cfg.frame_ms
        if utt_ms >= self.cfg.max_utterance_ms:
            # Force end to prevent runaway
            return self._end_utterance()

        if is_speech:
            self._silence_ms = 0
        else:
            self._silence_ms += self.cfg.frame_ms
            if self._silence_ms >= self.cfg.end_silence_ms:
                return self._end_utterance()

        return SegmentEvent(type=SegmentEventType.NONE, t_ms=self._t_ms)


# -----------------------------
# Quick standalone test
# -----------------------------
if __name__ == "__main__":
    cfg = SegmenterConfig(
        sample_rate=16000,
        frame_ms=20,
        start_trigger_frames=2,
        end_silence_ms=800,
        store_utterance_audio=False,
    )
    seg = SpeechSegmenter(cfg)

    # Simulate VAD decisions: silence -> speech -> silence
    decisions = [False]*10 + [True]*5 + [False]*50

    # Fake frames
    fake_frame = np.zeros(int(cfg.sample_rate * cfg.frame_ms / 1000), dtype=np.float32)

    for i, d in enumerate(decisions):
        ev = seg.update(fake_frame, d)
        if ev.type == SegmentEventType.SPEECH_START:
            print(f"[{i}] START at t={ev.t_ms}ms")
        elif ev.type == SegmentEventType.SPEECH_END:
            print(f"[{i}] END at t={ev.t_ms}ms, frames={ev.frames}")
