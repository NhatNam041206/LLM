# src/tts/audio_player.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass
class AudioPlayerConfig:
    output_device: Optional[int] = None  # sounddevice output device index
    volume: float = 1.0                  # simple volume scaling


class AudioPlayer:
    """
    Plays float32 mono audio using sounddevice.
    """

    def __init__(self, cfg: AudioPlayerConfig):
        self.cfg = cfg

    @staticmethod
    def list_output_devices():
        import sounddevice as sd
        devs = sd.query_devices()
        out = []
        for i, d in enumerate(devs):
            if d.get("max_output_channels", 0) > 0:
                out.append({"index": i, "name": d.get("name"), "max_out_ch": d.get("max_output_channels")})
        return out

    def play(self, audio: np.ndarray, sample_rate: int):
        import sounddevice as sd

        if audio is None or audio.size == 0:
            return

        if audio.ndim != 1:
            raise ValueError("AudioPlayer expects mono 1D audio")

        x = audio.astype(np.float32, copy=False)

        # volume (avoid clipping)
        vol = float(self.cfg.volume)
        if vol != 1.0:
            x = x * vol
            x = np.clip(x, -1.0, 1.0)

        sd.play(x, samplerate=int(sample_rate), device=self.cfg.output_device)
        sd.wait()
