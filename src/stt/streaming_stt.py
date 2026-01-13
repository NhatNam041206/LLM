"""
Streaming STT coordinator (Option B): mic -> align -> VAD -> segmenter -> ring buffer -> STT.

Emits:
- on_partial(text)
- on_final(text)

This module is designed to be used standalone (for testing) or integrated later
into an orchestrator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from audio.mic_input import MicInput
from audio.frame_aligner import FrameAligner
from audio.ring_buffer import AudioRingBuffer

from stt.vad import WebRTCVAD
from stt.segmenter import SpeechSegmenter, SegmenterConfig, SegmentEventType
from stt.stt_engine import STTEngine, STTEngineConfig


@dataclass
class StreamingSTTConfig:
    # Audio
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 20
    backend: str = "sounddevice"
    device: Optional[int] = None

    # VAD / segmentation
    vad_aggressiveness: int = 2
    start_trigger_frames: int = 2
    end_silence_ms: int = 800
    min_utterance_ms: int = 200

    # Rolling window
    rolling_window_sec: float = 8.0

    # STT inference
    infer_interval_ms: int = 400         # run STT every X ms while speaking
    stt_model: str = "tiny"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_beam_size: int = 1
    stt_language: Optional[str] = "en"

    # Output behavior
    emit_partials: bool = True
    partial_min_chars_delta: int = 1     # only emit partial if changed by at least N chars
    suppress_empty: bool = True          # don't emit empty partial/final


class StreamingSTT:
    def __init__(
        self,
        cfg: StreamingSTTConfig,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
    ):
        self.cfg = cfg
        self.on_partial = on_partial
        self.on_final = on_final

        self._running = False

        # Core components
        self.mic = MicInput(
            sample_rate=cfg.sample_rate,
            channels=cfg.channels,
            frame_ms=cfg.frame_ms,
            backend=cfg.backend,
            device=cfg.device,
            dtype="float32",
        )

        self.frame_size = int(cfg.sample_rate * cfg.frame_ms / 1000)
        self.aligner = FrameAligner(self.frame_size)

        self.vad = WebRTCVAD(
            sample_rate=cfg.sample_rate,
            frame_ms=cfg.frame_ms,
            aggressiveness=cfg.vad_aggressiveness,
        )

        self.segmenter = SpeechSegmenter(
            SegmenterConfig(
                sample_rate=cfg.sample_rate,
                frame_ms=cfg.frame_ms,
                start_trigger_frames=cfg.start_trigger_frames,
                end_silence_ms=cfg.end_silence_ms,
                min_utterance_ms=cfg.min_utterance_ms,
                store_utterance_audio=False,  # we use ring buffer instead
            )
        )

        self.ring = AudioRingBuffer(
            sample_rate=cfg.sample_rate,
            window_sec=cfg.rolling_window_sec,
        )

        self.stt = STTEngine(
            STTEngineConfig(
                model_size=cfg.stt_model,
                device=cfg.stt_device,
                compute_type=cfg.stt_compute_type,
                beam_size=cfg.stt_beam_size,
                language=cfg.stt_language,
                vad_filter=False,  # we already do VAD
            )
        )

        self._last_infer_t = 0.0
        self._last_partial = ""

    def start(self):
        if self._running:
            return
        self._running = True
        self._last_infer_t = 0.0
        self._last_partial = ""
        self.ring.clear()
        self.segmenter.reset()
        self.mic.start()

    def stop(self):
        self._running = False
        self.mic.stop()

    def run_forever(self):
        """
        Blocking loop: reads mic frames and produces partial/final transcripts.
        """
        self.start()
        try:
            for chunk in self.mic.frames():
                if not self._running:
                    break

                # Push chunk into aligner (chunk size may vary)
                self.aligner.push(chunk)

                # Process all full frames available
                while True:
                    frame = self.aligner.pop()
                    if frame is None:
                        break

                    # VAD decision for this frame
                    is_speech = self.vad.is_speech(frame)

                    # Segmenter state update (utterance boundaries)
                    ev = self.segmenter.update(frame, is_speech)

                    # Always append to rolling buffer while in speech OR near speech
                    # (Appending always is fine too; rolling window keeps last N sec)
                    self.ring.append(frame)

                    # If speaking, periodically run STT on rolling window for partials
                    now = time.time()
                    if self.cfg.emit_partials and self.segmenter.in_speech:
                        if (now - self._last_infer_t) * 1000.0 >= self.cfg.infer_interval_ms:
                            self._last_infer_t = now
                            window = self.ring.get(pad_to_full=False)
                            text = self.stt.transcribe(window)

                            if self.cfg.suppress_empty and not text:
                                continue

                            # emit only if changed enough
                            if abs(len(text) - len(self._last_partial)) >= self.cfg.partial_min_chars_delta or text != self._last_partial:
                                self._last_partial = text
                                if self.on_partial:
                                    self.on_partial(text)

                    # On utterance end: run a final transcription (fresh) and emit final
                    if ev.type == SegmentEventType.SPEECH_END:
                        window = self.ring.get(pad_to_full=False)
                        final_text = self.stt.transcribe(window)

                        if self.cfg.suppress_empty and not final_text:
                            self._last_partial = ""
                            continue

                        self._last_partial = ""
                        if self.on_final:
                            self.on_final(final_text)

        finally:
            self.stop()
