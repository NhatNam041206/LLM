#!/usr/bin/env python3
"""
Audio Calibration Tool

Helps you find optimal settings for:
1. Microphone input volume
2. RMS/Energy threshold for noise gate
3. VAD aggressiveness
4. Other audio parameters

This tool will:
- Measure silence (noise floor)
- Measure your speech
- Recommend optimal threshold values
- Generate .env settings
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import time
from collections import deque

from audio.mic_input import MicInput


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS (Root Mean Square) energy of audio."""
    return np.sqrt(np.mean(audio ** 2))


def calibrate_microphone():
    """Interactive calibration wizard."""
    
    print("=" * 70)
    print(" " * 20 + "AUDIO CALIBRATION WIZARD")
    print("=" * 70)
    print()
    print("This tool will help you find optimal audio settings for VAD.")
    print()
    
    # Configuration
    SAMPLE_RATE = 16000
    FRAME_MS = 20
    FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
    
    print(f"Configuration:")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    print(f"  Frame Size: {FRAME_MS} ms ({FRAME_SAMPLES} samples)")
    print()
    
    # Initialize microphone
    print("Initializing microphone...")
    mic = MicInput(
        sample_rate=SAMPLE_RATE,
        channels=1,
        frame_ms=FRAME_MS,
        backend="sounddevice",
        dtype="float32",
    )
    mic.start()
    print("✓ Microphone started")
    print()
    
    # ========================================
    # STEP 1: Measure silence (noise floor)
    # ========================================
    print("=" * 70)
    print("STEP 1: Measuring Silence (Noise Floor)")
    print("=" * 70)
    print()
    print("Please BE SILENT for 5 seconds.")
    print("We'll measure background noise from your environment.")
    print()
    input("Press ENTER when ready to start...")
    print()
    
    silence_rms_values = []
    silence_duration = 5  # seconds
    frames_needed = int(silence_duration * 1000 / FRAME_MS)
    
    print(f"Recording silence for {silence_duration} seconds...")
    print("BE QUIET! Don't talk, don't move...")
    print()
    
    frame_count = 0
    for chunk in mic.frames():
        if frame_count >= frames_needed:
            break
        
        rms = compute_rms(chunk)
        silence_rms_values.append(rms)
        
        # Visual progress
        bar_length = 40
        progress = frame_count / frames_needed
        filled = int(progress * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r[{bar}] {progress*100:.0f}%  RMS: {rms:.6f}", end="", flush=True)
        
        frame_count += 1
    
    print("\n")
    
    # Analyze silence
    silence_rms_array = np.array(silence_rms_values)
    silence_mean = np.mean(silence_rms_array)
    silence_max = np.max(silence_rms_array)
    silence_95percentile = np.percentile(silence_rms_array, 95)
    
    print("✓ Silence measurement complete")
    print()
    print("Silence Statistics:")
    print(f"  Mean RMS:  {silence_mean:.6f}")
    print(f"  Max RMS:   {silence_max:.6f}")
    print(f"  95th percentile: {silence_95percentile:.6f}")
    print()
    
    # ========================================
    # STEP 2: Measure speech
    # ========================================
    print("=" * 70)
    print("STEP 2: Measuring Your Speech")
    print("=" * 70)
    print()
    print("Please SPEAK NORMALLY for 10 seconds.")
    print("Talk as you would during normal conversation.")
    print("Example: Count from 1 to 20, say the alphabet, etc.")
    print()
    input("Press ENTER when ready to start...")
    print()
    
    speech_rms_values = []
    speech_duration = 10  # seconds
    frames_needed = int(speech_duration * 1000 / FRAME_MS)
    
    print(f"Recording speech for {speech_duration} seconds...")
    print("START TALKING NOW!")
    print()
    
    frame_count = 0
    for chunk in mic.frames():
        if frame_count >= frames_needed:
            break
        
        rms = compute_rms(chunk)
        speech_rms_values.append(rms)
        
        # Visual progress
        bar_length = 40
        progress = frame_count / frames_needed
        filled = int(progress * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Color code if speech detected
        if rms > silence_95percentile * 2:
            status = "\033[92m🎤 SPEECH\033[0m"
        else:
            status = "\033[90m... silence\033[0m"
        
        print(f"\r[{bar}] {progress*100:.0f}%  RMS: {rms:.6f}  {status}", end="", flush=True)
        
        frame_count += 1
    
    print("\n")
    
    # Analyze speech
    speech_rms_array = np.array(speech_rms_values)
    speech_mean = np.mean(speech_rms_array)
    speech_min = np.min(speech_rms_array)
    speech_05percentile = np.percentile(speech_rms_array, 5)
    speech_median = np.median(speech_rms_array)
    
    print("✓ Speech measurement complete")
    print()
    print("Speech Statistics:")
    print(f"  Mean RMS:  {speech_mean:.6f}")
    print(f"  Median RMS: {speech_median:.6f}")
    print(f"  Min RMS:   {speech_min:.6f}")
    print(f"  5th percentile: {speech_05percentile:.6f}")
    print()
    
    # Stop microphone
    mic.stop()
    
    # ========================================
    # STEP 3: Analysis and Recommendations
    # ========================================
    print("=" * 70)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    # Calculate separation
    separation_ratio = speech_05percentile / silence_95percentile
    
    print(f"Speech-to-Noise Separation: {separation_ratio:.2f}x")
    print()
    
    if separation_ratio < 1.5:
        print("⚠️  WARNING: Poor speech-to-noise separation!")
        print("   Your microphone is picking up too much background noise,")
        print("   or your speech is too quiet.")
        print()
        print("   Recommendations:")
        print("   1. Increase microphone input volume in system settings")
        print("   2. Move closer to the microphone")
        print("   3. Reduce background noise (turn off fans, close windows)")
        print()
    elif separation_ratio < 3:
        print("⚠️  Moderate separation - may have occasional false positives")
        print()
    else:
        print("✓ Good speech-to-noise separation!")
        print()
    
    # Recommend threshold
    # Use value between silence max and speech min
    recommended_threshold = silence_95percentile * 1.5
    safe_threshold = min(recommended_threshold, speech_05percentile * 0.8)
    
    print("Recommended Energy Threshold:")
    print(f"  Conservative (fewer false positives): {silence_95percentile * 2:.6f}")
    print(f"  Balanced: {safe_threshold:.6f}  ← RECOMMENDED")
    print(f"  Aggressive (catch more speech): {silence_95percentile * 1.2:.6f}")
    print()
    
    # VAD aggressiveness recommendation
    if separation_ratio > 4:
        recommended_agg = 0
        print("Recommended VAD_AGGRESSIVENESS: 0 (least sensitive)")
        print("  You have good audio quality - use less sensitive VAD")
    elif separation_ratio > 2.5:
        recommended_agg = 1
        print("Recommended VAD_AGGRESSIVENESS: 1")
        print("  Balanced setting for your audio environment")
    else:
        recommended_agg = 2
        print("Recommended VAD_AGGRESSIVENESS: 2")
        print("  More sensitive to compensate for noise")
    print()
    
    # ========================================
    # STEP 4: Generate .env settings
    # ========================================
    print("=" * 70)
    print("RECOMMENDED .ENV SETTINGS")
    print("=" * 70)
    print()
    print("Add these to your .env file:")
    print()
    print(f"# Audio Calibration Results (generated {time.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"VAD_ENERGY_THRESHOLD={safe_threshold:.6f}")
    print(f"VAD_AGGRESSIVENESS={recommended_agg}")
    print("VAD_SMOOTH_WINDOW=5")
    print("START_TRIGGER_FRAMES=3")
    print("VAD_SILENCE_MS=700")
    print()
    print("# Environment Analysis:")
    print(f"# Noise floor (95th): {silence_95percentile:.6f}")
    print(f"# Speech min (5th):   {speech_05percentile:.6f}")
    print(f"# Separation ratio:   {separation_ratio:.2f}x")
    print()
    print("=" * 70)
    print()
    
    # Offer to save
    save = input("Save these settings to .env file? [y/N]: ").strip().lower()
    if save == 'y':
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        
        settings = f"""
# Audio Calibration Results (generated {time.strftime('%Y-%m-%d %H:%M:%S')})
VAD_ENERGY_THRESHOLD={safe_threshold:.6f}
VAD_AGGRESSIVENESS={recommended_agg}

# Environment Analysis:
# Noise floor (95th): {silence_95percentile:.6f}
# Speech min (5th):   {speech_05percentile:.6f}
# Separation ratio:   {separation_ratio:.2f}x
"""
        
        print()
        print(f"Settings will be appended to: {env_path}")
        print("Review and adjust manually if needed.")
        print()
        
        with open(env_path, 'a') as f:
            f.write("\n" + settings)
        
        print("✓ Settings saved!")
    
    print()
    print("Calibration complete!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        calibrate_microphone()
    except KeyboardInterrupt:
        print("\n\nCalibration cancelled.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
