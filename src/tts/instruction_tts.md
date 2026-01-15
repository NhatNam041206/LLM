# INSTRUCTIONS — TTS Process 
**(Local Piper-based TTS)**

Goal
----
Provide a local TTS engine that converts short text replies into playable PCM audio and exposes a simple API for playback and integration with the orchestrator.

What this repo implements
-------------------------
- `PiperTTSEngine` in `src/tts/tts_engine.py` — wrapper around `piper.voice.PiperVoice`.
- `AudioPlayer` in `src/tts/audio_player.py` — plays float32 mono audio using `sounddevice`.

Core API
--------
- `PiperTTSEngine(PiperTTSConfig)` — constructor loads the Piper voice model once.
- `synthesize(text: str) -> (np.ndarray, int)` — returns `(audio_float32, sample_rate)`.
- `speak(text: str) -> None` — synthesizes and plays audio (calls `AudioPlayer.play`).
- `list_output_devices()` — convenience to list available sounddevice output devices.

Config (see `src/tts/tts_engine.py`)
----------------------------------
- `model_path` (str): path to Piper ONNX model file (e.g., `en_US-amy-medium.onnx`).
- `speaker_id`, `length_scale`, `noise_scale`, `noise_w_scale` — voice synthesis parameters.
- `output_device`, `volume` — playback settings forwarded to `AudioPlayer`.

Chunk handling and audio format
------------------------------
- `PiperVoice.synthesize()` yields `AudioChunk` objects with `audio_float_array` (float32 in [-1,1]).
- `PiperTTSEngine.synthesize()` collects chunk payloads, converts them to float32 mono arrays, concatenates and returns a single 1D float32 array.
- The engine attempts to detect a variety of chunk attributes (`audio_float_array`, `audio_int16_bytes`, `audio_int16_array`) and falls back to `np.asarray()` where possible.

Playback behavior and robustness
--------------------------------
- `AudioPlayer.play()` uses `sounddevice.play()` and `sd.wait()` — playback is blocking.
- To avoid last-word cut-off on some backends, the engine appends a short trailing silence pad (configurable in code) before returning audio.

Usage example
-------------
```
from tts.tts_engine import PiperTTSEngine, PiperTTSConfig

cfg = PiperTTSConfig(model_path='models/tts/en_US-amy-medium.onnx')
tts = PiperTTSEngine(cfg)
audio, sr = tts.synthesize('Hello world')
tts.speak('Hello world')
```

Notes
-----
- Ensure the model JSON (e.g., `en_US-amy-medium.onnx.json`) is present next to the ONNX file.
- For headless environments or non-blocking playback, replace `AudioPlayer.play()` implementation or modify to stream to an output thread.
- If using a different TTS backend, adapt the chunk extraction logic to the backend's audio field names.

Testing
-------
- `scripts/test_tts_once.py` demonstrates `speak()` and device listing.
- `scripts/test_tts_chatloop.py` provides an interactive loop for manual QA.

Troubleshooting
---------------
- If audio is silent: verify `model_path` and model JSON exist, and the voice loaded successfully.
- If words are cut: increase trailing pad or inspect playback pipeline for early termination.

References
----------
- Piper: `piper.voice.PiperVoice` AudioChunk API (float array + int16 helpers)
