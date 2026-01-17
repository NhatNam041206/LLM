# scripts/test_vad.py
import sys
import time
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.append("src")

from audio.mic_input import MicInput
from stt.vad import WebRTCVAD
from stt.segmenter import SpeechSegmenter, SegmenterConfig, SegmentEventType
from audio.frame_aligner import FrameAligner

def main():
    # ---- Load .env ----
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    # ---- Config ----
    sample_rate = 16000
    channels = 1
    frame_ms = 20
    backend = "sounddevice"     # switch to "pyaudio" if needed
    device = None               # set mic device index if needed

    # Read VAD config from .env
    vad_aggressiveness = int(os.getenv("VAD_AGGRESSIVENESS", "1"))
    vad_energy_threshold = float(os.getenv("VAD_ENERGY_THRESHOLD", "0.0"))
    vad_smooth_window = int(os.getenv("VAD_SMOOTH_WINDOW", "0"))
    end_silence_ms = int(os.getenv("VAD_SILENCE_MS", "1500"))
    start_trigger_frames = int(os.getenv("START_TRIGGER_FRAMES", "8"))
    min_utterance_ms = int(os.getenv("MIN_UTTERANCE_MS", "300"))
    max_utterance_ms = int(os.getenv("MAX_UTTERANCE_MS", "10000"))

    frame_size = int(sample_rate * frame_ms / 1000)
    aligner = FrameAligner(frame_size)

    mic = MicInput(
        sample_rate=sample_rate,
        channels=channels,
        frame_ms=frame_ms,
        backend=backend,
        device=device,
        dtype="float32"
    )

    vad = WebRTCVAD(
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        aggressiveness=vad_aggressiveness,
        energy_threshold=vad_energy_threshold
    )

    seg_cfg = SegmenterConfig(
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        start_trigger_frames=start_trigger_frames,
        end_silence_ms=end_silence_ms,
        min_utterance_ms=min_utterance_ms,
        max_utterance_ms=max_utterance_ms,
        store_utterance_audio=False,  # we only want boundaries here
    )
    segmenter = SpeechSegmenter(seg_cfg)

    print("=== VAD Streaming Test ===")
    print(f"Config from .env:")
    print(f"  sample_rate={sample_rate}, frame_ms={frame_ms}")
    print(f"  vad_aggressiveness={vad_aggressiveness}")
    print(f"  vad_energy_threshold={vad_energy_threshold}")
    print(f"  vad_smooth_window={vad_smooth_window}")
    print(f"  start_trigger_frames={start_trigger_frames}")
    print(f"  end_silence_ms={end_silence_ms}")
    print(f"  min_utterance_ms={min_utterance_ms}")
    print(f"  max_utterance_ms={max_utterance_ms}")
    print("\nTalk to trigger SPEECH START, stop to trigger SPEECH END.")
    print("Utterances longer than {:.1f}s will be automatically cut off.".format(max_utterance_ms / 1000))
    print("Ctrl+C to stop.\n")

    mic.start()

    last_state = None
    last_print_t = 0.0

    try:
        for frame in mic.frames():
            # Ensure mono
            if frame.ndim != 1:
                frame = frame.mean(axis=1).astype(np.float32)

            # VAD requires exact frame size
            # If mismatch occurs, your mic chunking isn't aligned -> fix in mic_input config
            aligner.push(frame)

            while True:
                fixed = aligner.pop()
                if fixed is None:
                    break

                is_speech = vad.is_speech(fixed)

                now = time.time()
                state = "SPEECH" if is_speech else "SILENCE"
                if state != last_state or (now - last_print_t) > 0.25:
                    print(f"\r{state:<7}", end="", flush=True)
                    last_state = state
                    last_print_t = now

                ev = segmenter.update(fixed, is_speech)
                if ev.type == SegmentEventType.SPEECH_START:
                    print(f"\n>>> SPEECH START (t={ev.t_ms}ms)")
                elif ev.type == SegmentEventType.SPEECH_END:
                    print(f">>> SPEECH END   (t={ev.t_ms}ms)  [silence>{end_silence_ms}ms]\n")

    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
