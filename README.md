**🎙️ Local Voice Assistant (STT → LLM → TTS)**

A fully local, real-time voice assistant pipeline that converts speech → text → response → speech, designed to run on affordable hardware (low RAM, optional GPU) with no API keys, no token limits, and full privacy.

**✨ Features**

🔊 Real-time Speech-to-Text (STT) using faster-whisper

🧠 Local LLM inference (1–2B or smaller models)

🗣️ Local Text-to-Speech (TTS) (interruptible)

🎧 Microphone streaming with VAD (Voice Activity Detection)

🔁 Turn-taking & barge-in support

🧩 Modular, extensible architecture

🔒 Fully offline – no cloud, no billing, no telemetry

**🧠 System Overview**

Microphone
   ↓
Audio Capture & Preprocessing
   ↓
VAD + Streaming Buffer
   ↓
STT (faster-whisper, int8)
   ↓
Dialogue Manager
   ↓
LLM (local small model)
   ↓
TTS (local)
   ↓
Speaker


*The system supports streaming partial transcripts, final utterance detection, and interrupting TTS when the user speaks.*

**🗂️ Project Structure**
```
voice-assistant/
├── README.md
├── src/
│   ├── main.py
│   ├── audio/
│   │   ├── mic_input.py
│   │   ├── preprocess.py
│   │   ├── ring_buffer.py
│   │   └── audio_output.py
│   ├── stt/
│   │   ├── vad.py
│   │   ├── stt_engine.py
│   │   ├── streaming_stt.py
│   │   └── text_stabilizer.py
│   ├── llm/
│   │   ├── llm_engine.py
│   │   ├── prompt_manager.py
│   │   └── memory.py
│   ├── tts/
│   │   ├── tts_engine.py
│   │   └── tts_queue.py
│   └── utils/
│       ├── logger.py
│       └── timing.py
│
└── scripts/
    ├── run_local.py
    ├── test_mic.py
    ├── test_stt.py
    ├── test_llm.py
    └── test_tts.py
```
**📦 Dependencies**
Core libraries
```
pip install \
  faster-whisper \
  sounddevice \
  pyaudio \
  numpy \
  soxr \
  webrtcvad-wheels
```

**Optional (recommended)**

torch – if using Silero VAD or certain TTS engines

llama-cpp-python – for local LLM inference

fastapi / websockets – if exposing a service

***Audio Requirements***

All audio is normalized to:
* Sample rate: 16,000 Hz
* Channels: Mono
* Frame size: 20–30 ms
* Format: PCM int16 or float32
* Resampling is handled automatically using soxr.

**STT Pipeline (Streaming Mode)**
* Microphone frames captured continuously
* VAD detects speech activity
* Rolling buffer (5–10 seconds)
* STT runs every 200–500 ms during speech

*Emits:*

Partial transcripts (live)
Final transcript after silence timeout
Recommended STT settings (low hardware)
Model: tiny or base
compute_type="int8"

beam_size=1

VAD aggressiveness: 2–3

**LLM Pipeline**
* Small local model (≤2B parameters)
* Short conversational memory (last N turns)
* Optimized for spoken responses
* No fine-tuning required for basic conversation
* Typical usage
* Triggered only after final STT text
* Optional token streaming
* Short, natural replies (voice-friendly)

**TTS Pipeline**
* Local TTS engine (e.g. Piper or equivalent)
* Sentence-level chunking
* Playback queue
* Immediate stop on barge-in
* Required TTS features
* speak(text)
* stop()
* is_speaking()

**💻 Hardware Targets**

* Designed to run on:

* 4–8 GB RAM

* CPU-only or low-end GPU

* Laptop / mini-PC / edge device

*Performance tips*
* Keep LLM context short
* Use quantized STT models
* Stream audio & text
* Avoid running all heavy tasks simultaneously

Author: Nam Nhat