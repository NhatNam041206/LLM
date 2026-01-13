"""
Simple microphone test script.

Prints:
- detected input devices
- actual stream sample rate
- basic RMS volume to confirm audio is real

Usage:
    python scripts/test_mic.py [--device DEVICE_ID] [--duration SECONDS]

Press Ctrl-C to stop early.
"""

import argparse
import sys
import time

import numpy as np
import sounddevice as sd


def list_input_devices():
    print("Detected input devices:")
    devices = sd.query_devices()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev.get('max_input_channels', 0) > 0:
            input_devices.append((i, dev))
            print(f"  [{i}] {dev['name']}  (channels={dev['max_input_channels']}, default_samplerate={int(dev['default_samplerate'])})")
    if not input_devices:
        print("  No input devices found.")
    return input_devices


def format_rms_bar(rms, width=30):
    # Normalize RMS roughly (0.0 - 0.3 typical for close mic); clamp
    level = min(max(rms / 0.3, 0.0), 1.0)
    filled = int(level * width)
    return "[" + ("#" * filled).ljust(width) + "]"


def main():
    parser = argparse.ArgumentParser(description="Test microphone and show RMS levels.")
    parser.add_argument('--device', type=int, default=None, help='Input device ID (index)')
    parser.add_argument('--out-device', type=int, default=None, help='Output device ID (index)')
    parser.add_argument('--duration', type=float, default=None, help='Duration to run in seconds (default: until Ctrl-C)')
    parser.add_argument('--channels', type=int, default=1, help='Number of input channels to read (default: 1)')
    parser.add_argument('--monitor', action='store_true', help='Play back incoming audio to the output device (monitor)')
    parser.add_argument('--gain', type=float, default=1.0, help='Playback gain when monitoring (default: 1.0)')
    args = parser.parse_args()

    input_devices = list_input_devices()

    # Choose input device
    device_id = args.device
    if device_id is None:
        try:
            default_dev = sd.default.device
            if default_dev is not None and isinstance(default_dev, (list, tuple)) and len(default_dev) >= 1 and default_dev[0] is not None:
                device_id = default_dev[0]
        except Exception:
            device_id = None

        if device_id is None:
            if input_devices:
                device_id = input_devices[0][0]
            else:
                print("No suitable input device available. Exiting.")
                sys.exit(1)

    try:
        dev_info = sd.query_devices(device_id, kind='input')
    except Exception as e:
        print(f"Failed to query device {device_id}: {e}")
        sys.exit(1)

    samplerate = int(dev_info.get('default_samplerate', 44100))
    in_channels = min(args.channels, int(dev_info.get('max_input_channels', 1)))

    out_device = args.out_device
    out_channels = 0
    if args.monitor:
        # choose output device
        if out_device is None:
            try:
                default_dev = sd.default.device
                if default_dev is not None and isinstance(default_dev, (list, tuple)) and len(default_dev) >= 2 and default_dev[1] is not None:
                    out_device = default_dev[1]
            except Exception:
                out_device = None

        if out_device is None:
            # fallback: pick first output-capable device
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev.get('max_output_channels', 0) > 0:
                    out_device = i
                    break

        if out_device is None:
            print("Monitoring requested but no output device found. Continuing without playback.")
            args.monitor = False
        else:
            try:
                out_info = sd.query_devices(out_device, kind='output')
                out_channels = int(out_info.get('max_output_channels', 1))
                print(f"\nUsing input device [{device_id}] {dev_info['name']}")
                print(f"Actual stream sample rate: {samplerate} Hz")
                print(f"Monitoring ON -> output device [{out_device}] {out_info['name']} (channels={out_channels})")
            except Exception as e:
                print(f"Failed to query output device {out_device}: {e}")
                args.monitor = False

    if not args.monitor:
        print(f"\nUsing device [{device_id}] {dev_info['name']}")
        print(f"Actual stream sample rate: {samplerate} Hz")
        print("Basic RMS volume (move your voice to see changes). Press Ctrl-C to stop.")
        print("✅ Output: you can talk and see RMS moving.")

    # Duplex stream callback (input->output) when monitoring, otherwise input-only RMS
    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"Stream status: {status}", file=sys.stderr)

        if indata.size == 0:
            if args.monitor:
                outdata.fill(0)
            return

        # Calculate RMS on the input (mean across channels)
        mono = np.mean(indata[:, :in_channels], axis=1) if in_channels > 1 else indata[:, 0]
        rms = float(np.sqrt(np.mean(mono ** 2)))
        bar = format_rms_bar(rms)
        print(f"\rRMS={rms:.5f} {bar}", end="", flush=True)

        if args.monitor:
            gain = float(args.gain)
            # Map input channels to output channels
            in_ch = indata.shape[1]
            out_ch = outdata.shape[1]

            # If same shape, copy with gain
            if in_ch == out_ch:
                outdata[:] = indata * gain
            else:
                # Mix or replicate channels to match output
                if out_ch > in_ch:
                    # replicate or pad
                    for ch in range(out_ch):
                        src = indata[:, ch] if ch < in_ch else indata[:, -1]
                        outdata[:, ch] = src * gain
                else:
                    # fewer output channels: average input to mono then copy
                    mono = np.mean(indata, axis=1)
                    for ch in range(out_ch):
                        outdata[:, ch] = mono * gain

    try:
        if args.monitor:
            # Use duplex stream with explicit device tuple (input, output)
            devices = (device_id, out_device)
            # Choose a channel count that accommodates both devices
            channels = max(in_channels, out_channels)
            with sd.Stream(device=devices, samplerate=samplerate, channels=channels, dtype='float32', callback=callback):
                start = time.time()
                try:
                    while True:
                        time.sleep(0.1)
                        if args.duration is not None and (time.time() - start) >= args.duration:
                            break
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
        else:
            # Input-only: use callback signature without outdata
            def in_callback(indata, frames, time_info, status):
                # adapt to the same logic for RMS printing
                if status:
                    print(f"Stream status: {status}", file=sys.stderr)
                if indata.size == 0:
                    return
                mono = np.mean(indata[:, :in_channels], axis=1) if in_channels > 1 else indata[:, 0]
                rms = float(np.sqrt(np.mean(mono ** 2)))
                bar = format_rms_bar(rms)
                print(f"\rRMS={rms:.5f} {bar}", end="", flush=True)

            with sd.InputStream(device=device_id, channels=in_channels, samplerate=samplerate, dtype='float32', callback=in_callback):
                start = time.time()
                try:
                    while True:
                        time.sleep(0.1)
                        if args.duration is not None and (time.time() - start) >= args.duration:
                            break
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
    except Exception as e:
        print(f"Error opening stream: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
