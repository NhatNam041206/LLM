import sys
import time

sys.path.append("src")

from stt.streaming_stt import StreamingSTT, StreamingSTTConfig


def on_partial(text: str):
    # Overwrite same line for partials
    print("\rPARTIAL: " + text[-120:], end="", flush=True)


def on_final(text: str):
    print("\nFINAL:   " + text)
    print("-" * 60)


def main():
    cfg = StreamingSTTConfig(
        sample_rate=16000,
        channels=1,
        frame_ms=20,
        backend="sounddevice",     # change to "pyaudio" if needed
        device=None,

        vad_aggressiveness=3,
        start_trigger_frames=3,
        end_silence_ms=1500,

        rolling_window_sec=8.0,
        infer_interval_ms=400,

        stt_model="tiny",
        stt_device="cpu",
        stt_compute_type="int8",
        stt_beam_size=3,
        stt_language="en",
    )

    stt = StreamingSTT(cfg, on_partial=on_partial, on_final=on_final)

    print("Streaming STT started. Speak into the mic.")
    print("Ctrl+C to stop.\n")
    try:
        stt.run_forever()
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    main()
