# INSTRUCTIONS — STT Process
**(Mic Streaming → Partial → Final)**

Goal
----
Provide a reliable streaming STT pipeline that emits partial transcripts during speech and a final transcript when an utterance ends.

What this repo implements
-------------------------
- `STTEngine` — a thin `faster-whisper` wrapper (`src/stt/stt_engine.py`) with `transcribe(audio: np.ndarray) -> str`.
- `StreamingSTT` — coordinator combining `MicInput`, `FrameAligner`, `WebRTCVAD`, `SpeechSegmenter`, `AudioRingBuffer` and `STTEngine` (`src/stt/streaming_stt.py`).

Output contract (events)
------------------------
- `on_partial(text: str)` — streaming, may change while user speaks
- `on_final(text: str)` — final stable transcript for the utterance

Controls
--------
- `start()`, `stop()`, and `run_forever()` on `StreamingSTT`.

Audio standard
--------------
- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Frame size: 20 ms
- Internal dtype: `float32` (convert to `int16` only when VAD requires it)

Key modules and responsibilities
-------------------------------
- `audio/mic_input.py` — capture microphone, produce aligned frames (used by `StreamingSTT`).
- `audio/frame_aligner.py` — assemble incoming chunks into fixed-size frames.
- `audio/ring_buffer.py` — rolling window of last N seconds (used as STT input window).
- `stt/vad.py` — `WebRTCVAD` wrapper; expects 20ms frames; set aggressiveness 0–3 (default 2).
- `stt/segmenter.py` — `SpeechSegmenter` drives utterance boundaries (start/end rules, silence timeout).
- `stt/stt_engine.py` — `STTEngine` loads Whisper model once and exposes `transcribe(audio)`.
- `stt/streaming_stt.py` — `StreamingSTTConfig` exposes settings (sample rate, frame_ms, rolling_window_sec, infer_interval_ms, stt model/config, vad params, emit_partials, etc.).

Important runtime behaviors (from `StreamingSTT`)
----------------------------------------------
- Microphone frames are aligned into fixed 20ms frames via `FrameAligner`.
- Each frame is VAD-checked and pushed to the ring buffer.
- While `segmenter.in_speech` is true, STT runs every `infer_interval_ms` (default 400ms) on the rolling buffer and emits partials.
- On `SegmentEventType.SPEECH_END`, a final STT pass runs on the rolling buffer and emits `on_final`.
- `emit_partials` and `suppress_empty` control whether partials/finals are emitted when empty.

Recommended settings for low-resource machines
---------------------------------------------
- `rolling_window_sec`: 6–8s
- `infer_interval_ms`: 300–500ms
- STT model: `tiny` or `base` with `compute_type='int8'`
- VAD aggressiveness: 2

Testing and integration
-----------------------
- Run `scripts/test_mic.py` to verify microphone capture.
- Use `StreamingSTT` with simple callbacks to print partials/finals and tune `infer_interval_ms` and `rolling_window_sec`.

Usage example (pseudo)
-----------------------
```
from stt.streaming_stt import StreamingSTT, StreamingSTTConfig

def on_partial(t): print('partial:', t)
def on_final(t): print('final:', t)

cfg = StreamingSTTConfig()
stt = StreamingSTT(cfg, on_partial=on_partial, on_final=on_final)
stt.run_forever()
```

Notes
-----
- Keep STT calls rate-limited (do not transcribe every audio frame).
- Use `ring.get(pad_to_full=False)` to avoid blocking padding of the window when running STT.