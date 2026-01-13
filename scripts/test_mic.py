# scripts/test_mic.py
import sys
import time
import numpy as np

# Allow "from audio..." imports
sys.path.append("src")

from audio.mic_input import MicInput


def list_devices():
    print("\n=== sounddevice input devices ===")
    try:
        devs = MicInput.list_input_devices_sounddevice()
        for d in devs:
            print(f"[{d['index']}] {d['name']} (max_in_ch={d['max_input_channels']})")
    except Exception as e:
        print("sounddevice device listing failed:", e)

    print("\n=== pyaudio input devices ===")
    try:
        devs = MicInput.list_input_devices_pyaudio()
        for d in devs:
            print(f"[{d['index']}] {d['name']} (max_in_ch={d['max_input_channels']} sr={d['default_sample_rate']})")
    except Exception as e:
        print("pyaudio device listing failed:", e)


def main():
    list_devices()

    # ---- Config ----
    sample_rate = 16000
    channels = 1
    frame_ms = 20
    backend = "sounddevice"  # switch to "pyaudio" if needed
    device = None            # set an integer index if you want a specific mic

    mic = MicInput(
        sample_rate=sample_rate,
        channels=channels,
        frame_ms=frame_ms,
        backend=backend,
        device=device,
        dtype="float32"
    )

    # Optional: enable live loopback (hear your voice)
    loopback = True

    out_stream = None
    if loopback:
        try:
            import sounddevice as sd
            out_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
                blocksize=int(sample_rate * frame_ms / 1000),
            )
            out_stream.start()
        except Exception as e:
            print("Loopback audio output failed (continuing without loopback):", e)
            out_stream = None

    mic.start()
    print("\nListening... (Ctrl+C to stop)")
    print(f"Configured sample_rate={sample_rate}, channels={channels}, frame_ms={frame_ms}, backend={backend}")

    # Try to print actual stream samplerate if using sounddevice backend
    try:
        sd_stream = getattr(mic, "_sd_stream", None)
        if sd_stream is not None:
            print(f"Actual sounddevice stream sample_rate={sd_stream.samplerate}")
    except Exception:
        pass

    try:
        for frame in mic.frames():
            # frame: float32 mono (n_samples,)
            rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0

            # Simple RMS bar
            bar_len = min(50, int(rms * 200))  # adjust gain for display
            bar = "#" * bar_len

            print(f"\rRMS={rms:.4f} |{bar:<50}", end="", flush=True)

            # Loopback: play what we record (hear yourself)
            if out_stream is not None:
                if channels == 1:
                    out_stream.write(frame.reshape(-1, 1))
                else:
                    out_stream.write(frame)

            time.sleep(0.0)

    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        if out_stream is not None:
            try:
                out_stream.stop()
                out_stream.close()
            except Exception:
                pass
        print("\nStopped.")


if __name__ == "__main__":
    main()
