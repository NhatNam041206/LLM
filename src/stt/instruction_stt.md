# INSTRUCTIONS — STT Process 
**(Mic Streaming → Partial → Final)**

## Goal

Build a local STT module that:

* Takes live microphone audio

* Uses VAD to detect speech

* Produces partial transcript while speaking (streaming)

* Produces final transcript when the utterance ends (silence timeout)

Available libraries:
```
faster_whisper, sounddevice, pyaudio, numpy, soxr, webrtcvad-wheels.
```
## Output contract (what STT must provide)

STT module must expose:

* Events / Callbacks

* on_partial(text: str) — streaming updates (can change)

* on_final(text: str) — final text for this utterance (stable)

Controls
```
start()
stop()
```
Orchestrator/LLM/TTS plugs into these events.

## Required Audio Standard
All internal processing should use:
```
Sample rate: 16000 Hz
Channels: 1 (mono)
Frame size: 20 ms (recommended)
dtype: float32 internally (or int16 if required by VAD)
```

Purpose:

Whisper-family models expect 16kHz mono; consistent framing makes VAD + timing reliable.

## STT Pipeline Overview

**Flow**
```
Mic capture → raw frames

Preprocess → downmix + resample → 16k mono

VAD → speech / silence decisions per frame

Segmenter → decides utterance boundaries

Rolling buffer → keep last N seconds of audio

STT loop → every X ms, transcribe rolling window while in speech

Stabilizer → commit stable prefix (reduce flicker)

Emit on_partial, on_final
```

## Module Breakdown

1) ```audio/mic_input.py```

**Responsibilities**

* List/select mic device
* Stream audio frames continuously
* Provide frames as float32 numpy arrays

**Requirements**

* Use either sounddevice OR pyaudio (pick one, don’t mix)
* Chunk size should align to FRAME_MS:
at 16kHz, 20ms = 320 samples

**Test**

* Print RMS/energy each frame so you see voice activity.

2) ```audio/preprocess.py```

**Responsibilities**

Convert any input format to standard 16k mono:

* downmix stereo → mono
* resample → 16k using soxr

**Test**

* Verify output: shape is (N,), sample rate constant.

3) ```stt/vad.py (WebRTC VAD)```

**Responsibilities**
* Given 20ms frames, return is_speech: bool

**Requirements**

* WebRTC VAD expects 16-bit PCM bytes
* Convert float32 → int16 safely
* Aggressiveness: 0–3 (start with 2)

**Test**

* Print SPEECH / SILENCE live

4) `stt/segmenter` *(can live inside streaming_stt.py)*

**Responsibilities**
Maintain speech state using VAD outputs:

* in_speech boolean
* speech_start_time
* end utterance when silence lasts VAD_SILENCE_MS (e.g., 800ms)

Recommended rules

Start utterance after N consecutive speech frames (e.g., 2–3 frames)

End utterance after silence_ms continuously

Test

Print “START” and “END” boundaries accurately.

5) audio/ring_buffer.py

Responsibilities

Maintain rolling window of last ROLLING_WINDOW_SEC seconds

Support:

append(frame)

get() (returns newest window)

Recommended

6–10 seconds window (start with 8s)

6) stt/stt_engine.py (faster-whisper wrapper)

Responsibilities

Load model once

Provide transcribe(audio_array) returning text

Recommended config (low hardware)

model: tiny (upgrade to base later)

compute_type="int8" (CPU)

beam_size=1

language set if known (helps)

7) stt/text_stabilizer.py

Responsibilities
Streaming transcripts often “jump”. Stabilizer reduces flicker.

Simple effective method

Keep last partial string

Find longest common prefix (LCP) with new partial

Commit only when stable for k iterations (e.g., 2–3)

At utterance end: output final full text

Test

Partials should update smoothly, not rewrite everything every time.

8) stt/streaming_stt.py (the coordinator)

Responsibilities
Main STT loop:

Consume preprocessed frames

VAD gate + segmenter

Update ring buffer

Run transcribe every STT_INFER_INTERVAL_MS while in_speech

Emit partials + final

Important

Do NOT transcribe on every frame (too slow)

Run STT on a timer (200–500ms)

Only run while speech detected