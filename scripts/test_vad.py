# scripts/test_vad.py
import sys
import time
import numpy as np

sys.path.append("src")

from audio.mic_input import MicInput
from stt.vad import WebRTCVAD
from stt.segmenter import SpeechSegmenter, SegmenterConfig, SegmentEventType
from audio.frame_aligner import FrameAligner

def main():
    # ---- Config ----
    sample_rate = 16000
    channels = 1
    frame_ms = 20
    backend = "sounddevice"     # switch to "pyaudio" if needed
    device = None               # set mic device index if needed

    vad_aggressiveness = 3
    end_silence_ms = 1500
    start_trigger_frames = 3

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
        aggressiveness=vad_aggressiveness
    )

    seg_cfg = SegmenterConfig(
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        start_trigger_frames=start_trigger_frames,
        end_silence_ms=end_silence_ms,
        min_utterance_ms=200,
        store_utterance_audio=False,  # we only want boundaries here
    )
    segmenter = SpeechSegmenter(seg_cfg)

    print("=== VAD Streaming Test ===")
    print(f"sample_rate={sample_rate}, frame_ms={frame_ms}, vad_aggr={vad_aggressiveness}, end_silence_ms={end_silence_ms}")
    print("Talk to trigger SPEECH START, stop to trigger SPEECH END.")
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
