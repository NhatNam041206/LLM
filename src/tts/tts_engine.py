# src/tts/tts_engine.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from tts.text_normalizer import TextNormalizer
from tts.audio_player import AudioPlayer, AudioPlayerConfig


@dataclass
class PiperTTSConfig:
    model_path: str

    speaker_id: int = 0
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8

    output_device: Optional[int] = None
    volume: float = 1.0



class PiperTTSEngine:
    """
    Piper TTS engine for API:
      PiperVoice.synthesize(text, syn_config) -> Iterable[AudioChunk]
    """

    def __init__(self, cfg: PiperTTSConfig):
        self.cfg = cfg

        if not os.path.exists(cfg.model_path):
            raise FileNotFoundError(f"TTS model not found: {cfg.model_path}")

        from piper.voice import PiperVoice
        from piper.config import SynthesisConfig

        self.SynthesisConfig = SynthesisConfig

        self._normalizer = TextNormalizer()
        self._player = AudioPlayer(AudioPlayerConfig(output_device=cfg.output_device, volume=cfg.volume))

        # Load ONCE
        self.voice = PiperVoice.load(cfg.model_path)

        # Read sample rate from voice config (common attribute name)
        # Some versions store it on voice.config.sample_rate
        self.sample_rate = int(getattr(getattr(self.voice, "config", None), "sample_rate", 22050))

    def _make_syn_config(self):
        return self.SynthesisConfig(
            speaker_id=self.cfg.speaker_id,
            length_scale=self.cfg.length_scale,
            noise_scale=self.cfg.noise_scale,
            noise_w_scale=self.cfg.noise_w_scale,
            normalize_audio=True,
            volume=1.0,
        )

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        text = self._normalizer.normalize(text)
        if not text:
            return np.zeros(0, dtype=np.float32), self.sample_rate

        syn_config = self._make_syn_config()

        chunks = self.voice.synthesize(text, syn_config=syn_config)

        # Collect chunks -> int16 -> float32
        pcm_parts = []
        sr = self.sample_rate

        # Iterate the previously obtained chunks iterable
        for ch in chunks:
            # sample_rate may exist
            if hasattr(ch, "sample_rate") and getattr(ch, "sample_rate"):
                sr = int(ch.sample_rate)

            # 1) If chunk itself is bytes-like
            if isinstance(ch, (bytes, bytearray, memoryview)):
                arr_i16 = np.frombuffer(ch, dtype=np.int16)
                pcm_parts.append(arr_i16.astype(np.float32) / 32768.0)
                continue

            # 2) Try common attribute names that might hold PCM. Include
            #    attributes used by Piper's AudioChunk (audio_float_array, audio_int16_bytes/array).
            data = None
            for attr in (
                "audio_float_array",
                "audio",
                "pcm",
                "data",
                "samples",
                "chunk",
                "audio_int16_bytes",
                "audio_int16_array",
            ):
                if hasattr(ch, attr):
                    data = getattr(ch, attr)
                    break

            # 3) If still None, try __dict__ keys (some dataclasses store fields there)
            if data is None and hasattr(ch, "__dict__"):
                for k in (
                    "audio_float_array",
                    "audio",
                    "pcm",
                    "data",
                    "samples",
                    "audio_int16_bytes",
                    "audio_int16_array",
                ):
                    if k in ch.__dict__:
                        data = ch.__dict__[k]
                        break

            if data is None:
                # last resort: try converting chunk to numpy directly
                try:
                    arr = np.asarray(ch)
                    if arr.size:
                        if arr.dtype == np.int16:
                            pcm_parts.append(arr.astype(np.float32) / 32768.0)
                        elif np.issubdtype(arr.dtype, np.floating):
                            pcm_parts.append(arr.astype(np.float32, copy=False))
                    continue
                except Exception:
                    continue

            # Now convert `data` into float32 audio
            if isinstance(data, (bytes, bytearray, memoryview)):
                arr_i16 = np.frombuffer(data, dtype=np.int16)
                pcm_parts.append(arr_i16.astype(np.float32) / 32768.0)
            else:
                arr = np.asarray(data)
                if arr.size == 0:
                    continue
                if arr.dtype == np.int16:
                    pcm_parts.append(arr.astype(np.float32) / 32768.0)
                elif np.issubdtype(arr.dtype, np.floating):
                    pcm_parts.append(arr.astype(np.float32, copy=False))
                else:
                    # unknown dtype, attempt int16 interpretation and skip on failure
                    try:
                        pcm_parts.append(arr.astype(np.int16).astype(np.float32) / 32768.0)
                    except Exception:
                        continue


        if not pcm_parts:
            return np.zeros(0, dtype=np.float32), sr

        audio = np.concatenate(pcm_parts, axis=0).astype(np.float32, copy=False)

        # Ensure mono 1D
        if audio.ndim != 1:
            audio = audio.reshape(-1).astype(np.float32, copy=False)

        # Add small trailing silence to avoid last-word cut-off on some audio backends
        pad_ms = 50  # milliseconds
        pad_len = int(sr * pad_ms / 1000)
        if pad_len > 0:
            audio = np.concatenate([audio, np.zeros(pad_len, dtype=np.float32)])

        audio = np.clip(audio, -1.0, 1.0)
        return audio, sr

    def speak(self, text: str) -> None:
        audio, sr = self.synthesize(text)
        self._player.play(audio, sr)
        print("sr=", sr, "len=", len(audio), "max=", float(np.max(np.abs(audio))) if len(audio) else 0)


    def list_output_devices(self):
        return AudioPlayer.list_output_devices()
