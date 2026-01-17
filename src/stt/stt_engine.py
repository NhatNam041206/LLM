
"""
STT Engine wrapper around faster-whisper.

Responsibilities
----------------
- Load Whisper model once (expensive operation)
- Provide a simple `transcribe(audio)` method
- Hide faster-whisper / CTranslate2 details from the rest of the system

Design notes
------------
- This engine is *stateless* across calls (audio in → text out)
- Streaming logic (rolling window, partials, timing) lives elsewhere
- Safe to call repeatedly in a loop
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from faster_whisper import WhisperModel


@dataclass
class STTEngineConfig:
    model_size: str = "small"        # "tiny", "base", "small", ...
    device: str = "cpu"             # "cpu" or "cuda"
    compute_type: str = "int8"       # "int8", "int8_float16", "float16"
    beam_size: int = 1
    language: Optional[str] = "en"   # set None for auto-detect
    task: str = "transcribe"         # or "translate"
    vad_filter: bool = False         # usually False if you already do VAD
    initial_prompt: Optional[str] = None


class STTEngine:
    """
    Thin wrapper around faster-whisper WhisperModel.
    """

    def __init__(self, cfg: STTEngineConfig):
        self.cfg = cfg

        # Load model ONCE (this is expensive)
        self.model = WhisperModel(
            self.cfg.model_size,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe an audio array into text.

        Parameters
        ----------
        audio : np.ndarray
            Mono audio samples, float32, 16 kHz
            Shape: (n_samples,)

        Returns
        -------
        str
            Transcribed text (may be empty if no speech)
        """
        if audio is None:
            return ""

        # Ensure correct format
        if audio.ndim != 1:
            raise ValueError("STTEngine.transcribe expects mono 1D audio")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        # faster-whisper returns an iterator of segments
        segments, info = self.model.transcribe(
            audio,
            beam_size=self.cfg.beam_size,
            language=self.cfg.language,
            task=self.cfg.task,
            vad_filter=self.cfg.vad_filter,
            initial_prompt=self.cfg.initial_prompt,
        )

        # Concatenate text segments
        texts = []
        for seg in segments:
            if seg.text:
                texts.append(seg.text)

        return "".join(texts).strip()

    def warmup(self, seconds: float = 0.5) -> None:
        """
        Optional: run a short dummy inference to warm up the model
        (useful to avoid first-call latency).
        """
        n = int(seconds * 16000)
        dummy_audio = np.zeros(n, dtype=np.float32)
        _ = self.transcribe(dummy_audio)


# -----------------------------
# Quick standalone test
# -----------------------------
if __name__ == "__main__":
    cfg = STTEngineConfig(
        model_size="small",
        device="cpu",
        compute_type="int8",
        beam_size=1,
        language="en",
    )
    stt = STTEngine(cfg)
    stt.warmup()

    print("STT engine loaded successfully.")