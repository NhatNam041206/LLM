**🎙️ Local Voice Assistant (STT → LLM → TTS)**

A fully local, real-time voice assistant pipeline that converts speech → text → response → speech. It is designed to run on modest hardware (4–8 GB RAM) and keeps all processing offline.

✨ Features
- Streaming STT with VAD and rolling buffer (`src/stt/streaming_stt.py`)
- Local LLM inference via `llama-cpp-python` (`src/llm/llm_engine.py`)
- Local TTS via Piper (`src/tts/tts_engine.py`)
- Modular components: `audio/`, `stt/`, `llm/`, `tts/`

Project layout (key files)
```
./
├─ README.md
├─ src/
│  ├─ audio/
│  │  ├─ mic_input.py
│  │  ├─ frame_aligner.py
│  │  └─ ring_buffer.py
│  ├─ stt/
│  │  ├─ stt_engine.py
│  │  ├─ streaming_stt.py
│  │  └─ vad.py
│  ├─ llm/
│  │  ├─ llm_engine.py
│  │  └─ prompt_manager.py
│  └─ tts/
│     ├─ tts_engine.py
│     └─ audio_player.py
└─ scripts/
   ├─ test_tts_once.py
   ├─ test_tts_chatloop.py
   ├─ test_stt_streaming.py
   └─ test_llm.py
```

Dependencies
------------
Install required packages (core):
```bash
pip install -r requirements.txt
```
If installing manually, the main packages are:
```bash
pip install faster-whisper sounddevice numpy soxr webrtcvad-wheels
pip install llama-cpp-python   # for LLM via llama.cpp bindings
pip install piper             # if using Piper TTS package
```

Notes on optional/OS-specific deps
- `pyaudio` is optional; `sounddevice` is the primary playback/capture library used here.
- `torch` is only required for optional components (not used in main flow).

Audio requirements
------------------
- Internal standard: 16 kHz, mono, 20 ms frames, float32.
- STT components expect 16 kHz mono windows; resampling is handled automatically.

Model setup (Important!)
------------------------
**Note:** The `models/` folder is **not** published in this repository. You must download the required models manually.

**TTS Model (Piper)**
- Download from: https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/amy/medium
- Files needed: `en_US-amy-medium.onnx` and `en_US-amy-medium.onnx.json`
- Place in: `models/tts/`
- Usage: Update `model_path` in `scripts/test_tts_once.py` to point to the `.onnx` file

**LLM Model (Llama via llama.cpp)**
- Download from: https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF/tree/main
- File needed: `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (or similar GGUF quantization)
- Place in: `models/llm/` (or any accessible path)
- Usage: Update `model_path` in LLM config to point to the `.gguf` file

After downloading, your directory should look like:
```
./models/
├─ tts/
│  ├─ en_US-amy-medium.onnx
│  └─ en_US-amy-medium.onnx.json
└─ llm/
   └─ Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

Quick start
-----------
1. Ensure Python v3.9+ and required packages installed.
2. Point model paths in `scripts/test_tts_once.py` or other scripts.
3. Run example scripts:
```bash
python scripts/test_tts_once.py      # synth + playback example
python scripts/test_tts_chatloop.py # interactive TTS loop
python scripts/test_stt_streaming.py# streaming STT demo
```

Where to look next
------------------
- `src/stt/` — streaming coordinator, VAD, ring buffer
- `src/llm/` — prompt manager and simple LLM engine (llama.cpp backend)
- `src/tts/` — Piper TTS wrapper and player
- `src/tts/instruction_tts.md`, `src/stt/instruction_stt.md`, `src/llm/instruction_llm.md` — human-facing instructions for each subsystem

Troubleshooting
---------------
- If audio is silent for TTS: check `model_path` and presence of `.json` config file next to the ONNX model.
- If STT returns empty transcripts: confirm microphone device, sample rate, and VAD aggressiveness.

Author: Nam Nhat