import queue, numpy as np, sounddevice as sd
from faster_whisper import WhisperModel
import time
SAMPLE_RATE = 16000
CHUNK_SEC = 1.0
ROLLING_SEC = 6.0

audio_q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_q.put(indata.copy())

model = WhisperModel("medium", device="cuda", compute_type="float16")  # quantized

rolling = np.zeros(int(SAMPLE_RATE * ROLLING_SEC), dtype=np.float32)

with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype="float32", callback=callback):
    print("Listening...")
    while True:
        chunk = audio_q.get()
        chunk = chunk.reshape(-1)

        # rolling buffer update
        rolling = np.roll(rolling, -len(chunk))
        rolling[-len(chunk):] = chunk

        # transcribe the rolling window (chunked streaming style)
        segments, info = model.transcribe(
            rolling,
            beam_size=3,
            vad_filter=True,   # built-in VAD filter (helps)
            language="en"
        )

        text = ""
        probs = []
        for s in segments:
            text += s.text
            # Confidence is 1 - no_speech_prob
            confidence = 1 - s.no_speech_prob
            probs.append((s.text, confidence))
        
        text = text.strip()
        if text:
            print(f"Text: {text}")
            for seg_text, prob in probs:
                print(f"  - '{seg_text.strip()}': {prob:.4f}")
        time.sleep(0.1)
