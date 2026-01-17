"""
Streaming STT coordinator (Option B): mic -> align -> VAD -> segmenter -> ring buffer -> STT.

Emits:
- on_partial(text)
- on_final(text, audio_buffer)

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

from stt.vad import WebRTCVAD, VADSmoother
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
    vad_energy_threshold: float = 0.0    # RMS threshold (noise gate), 0.0 = disabled
    vad_smooth_window: int = 5           # frames to smooth VAD decisions (anti-flutter)
    start_trigger_frames: int = 2
    end_silence_ms: int = 800
    min_utterance_ms: int = 200
    max_utterance_ms: int = 10_000       # maximum speech duration (10 seconds)

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
    partial_min_chars_delta: int = 5     # only emit partial if changed by at least N chars (reduces noise)
    suppress_empty: bool = True          # don't emit empty partial/final
    
    # Debug
    debug_show_vad_state: bool = False   # show real-time VAD decisions


class StreamingSTT:
    def __init__(
        self,
        cfg: StreamingSTTConfig,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str, Optional[np.ndarray]], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
    ):
        self.cfg = cfg
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        self._running = False
        
        # Debug state tracking (only print when changed)
        self._last_debug_state = None
        self._last_debug_print = 0.0
        
        # Performance tracking
        self._frames_processed = 0
        self._partial_infer_times = []
        self._final_infer_times = []
        self._utterance_start_time = None

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
            energy_threshold=cfg.vad_energy_threshold,
        )
        
        # VAD smoother (anti-flutter)
        self.vad_smoother = VADSmoother(window_size=cfg.vad_smooth_window)

        self.segmenter = SpeechSegmenter(
            SegmenterConfig(
                sample_rate=cfg.sample_rate,
                frame_ms=cfg.frame_ms,
                start_trigger_frames=cfg.start_trigger_frames,
                end_silence_ms=cfg.end_silence_ms,
                min_utterance_ms=cfg.min_utterance_ms,
                max_utterance_ms=cfg.max_utterance_ms,
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
        self.vad_smoother.reset()
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

                    # VAD decision for this frame (raw)
                    is_speech_raw = self.vad.is_speech(frame)
                    
                    # Smooth the VAD decision (anti-flutter)
                    is_speech = self.vad_smoother.smooth(is_speech_raw)

                    # Segmenter state update (utterance boundaries)
                    ev = self.segmenter.update(frame, is_speech)
                    
                    # Debug: Show VAD state (only when changed or every 100ms to avoid slowdown)
                    if self.cfg.debug_show_vad_state and ev.type == SegmentEventType.NONE:
                        now = time.time()
                        state_symbol = "🟢 SPEECH" if is_speech else "🔴 SILENCE"
                        if self.segmenter.in_speech:
                            silence_ms = getattr(self.segmenter, '_silence_ms', 0)
                            # Round to nearest 20ms to reduce unique states
                            silence_rounded = (silence_ms // 20) * 20
                            current_state = (state_symbol, True, silence_rounded)
                        else:
                            current_state = (state_symbol, False, 0)
                        
                        # Only print if state changed OR 100ms elapsed (reduce print frequency)
                        if current_state != self._last_debug_state or (now - self._last_debug_print) >= 0.1:
                            if self.segmenter.in_speech:
                                print(f"\r{state_symbol} | In utterance (silence: {silence_ms}ms/{self.cfg.end_silence_ms}ms)", end="", flush=True)
                            else:
                                print(f"\r{state_symbol} | Waiting for speech...", end="", flush=True)
                            self._last_debug_state = current_state
                            self._last_debug_print = now
                    
                    # Trigger speech start callback
                    if ev.type == SegmentEventType.SPEECH_START:
                        if self.cfg.debug_show_vad_state:
                            print("\r\033[2K", end="", flush=True)  # Clear line
                        if self.on_speech_start:
                            self.on_speech_start()
                        # Track utterance start time
                        self._utterance_start_time = time.time()
                        self._frames_processed = 0

                    # Always append to rolling buffer while in speech OR near speech
                    # (Appending always is fine too; rolling window keeps last N sec)
                    self.ring.append(frame)
                    
                    # Count frames for performance tracking
                    if self.segmenter.in_speech:
                        self._frames_processed += 1

                    # If speaking, periodically run STT on rolling window for partials
                    now = time.time()
                    if self.cfg.emit_partials and self.segmenter.in_speech:
                        if (now - self._last_infer_t) * 1000.0 >= self.cfg.infer_interval_ms:
                            self._last_infer_t = now
                            
                            # Time the STT inference
                            infer_start = time.time()
                            window = self.ring.get(pad_to_full=False)
                            text = self.stt.transcribe(window)
                            infer_time = (time.time() - infer_start) * 1000  # ms
                            
                            # Track inference times (keep last 10)
                            self._partial_infer_times.append(infer_time)
                            if len(self._partial_infer_times) > 10:
                                self._partial_infer_times.pop(0)

                            if self.cfg.suppress_empty and not text:
                                continue

                            # emit only if changed enough
                            if abs(len(text) - len(self._last_partial)) >= self.cfg.partial_min_chars_delta or text != self._last_partial:
                                self._last_partial = text
                                if self.on_partial:
                                    self.on_partial(text)

                    # On utterance end: run a final transcription (fresh) and emit final
                    if ev.type == SegmentEventType.SPEECH_END:
                        # Clear VAD debug display
                        if self.cfg.debug_show_vad_state:
                            print("\r\033[2K", end="", flush=True)
                        
                        # Trigger speech end callback
                        if self.on_speech_end:
                            self.on_speech_end()
                        
                        # Time the final STT inference
                        final_start = time.time()
                        window = self.ring.get(pad_to_full=False)
                        final_text = self.stt.transcribe(window)
                        final_infer_time = (time.time() - final_start) * 1000  # ms
                        
                        # Calculate utterance statistics
                        if self._utterance_start_time:
                            utterance_duration = (time.time() - self._utterance_start_time) * 1000  # ms
                            audio_duration = self._frames_processed * self.cfg.frame_ms  # ms
                            
                            # Log performance metrics
                            if self.cfg.debug_show_pipeline_status or self.cfg.debug_show_vad_state:
                                avg_partial = sum(self._partial_infer_times) / len(self._partial_infer_times) if self._partial_infer_times else 0
                                print(f"\n📊 [PERF] Utterance: {audio_duration:.0f}ms audio, {utterance_duration:.0f}ms real-time", flush=True)
                                print(f"📊 [PERF] STT: {len(self._partial_infer_times)} partials (avg {avg_partial:.0f}ms), final {final_infer_time:.0f}ms", flush=True)
                            
                            # Reset for next utterance
                            self._partial_infer_times.clear()

                        if self.cfg.suppress_empty and not final_text:
                            self._last_partial = ""
                            continue

                        self._last_partial = ""
                        if self.on_final:
                            # Pass both text and audio buffer
                            self.on_final(final_text, window)
                        
                        # Clear ring buffer after final emission to prevent old audio accumulation
                        self.ring.clear()

        finally:
            self.stop()
