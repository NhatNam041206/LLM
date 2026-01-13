"""
Rolling audio ring buffer for streaming STT.

Responsibilities
----------------
- Maintain a rolling window of the last `window_sec` seconds of audio
- Support:
    - append(frame): add new samples (mono float32)
    - get(): return the newest window as a contiguous numpy array (float32)
- Designed for real-time "Option B" streaming (rolling window + periodic STT)

Notes
-----
- This buffer stores *samples*, not frames, so it works even if your input chunk sizes vary.
- If you already use a FrameAligner, you can append aligned frames (fixed size),
  but it's not required.

Recommended
-----------
- window_sec: 6–10 seconds (start with 8s)
- sample_rate: 16000
- dtype: float32
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RingBufferConfig:
    sample_rate: int = 16000
    window_sec: float = 8.0
    dtype: str = "float32"


class AudioRingBuffer:
    def __init__(self, sample_rate: int = 16000, window_sec: float = 8.0, dtype: str = "float32"):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if window_sec <= 0:
            raise ValueError("window_sec must be > 0")

        self.cfg = RingBufferConfig(sample_rate=sample_rate, window_sec=window_sec, dtype=dtype)
        self.capacity = int(round(self.cfg.sample_rate * self.cfg.window_sec))
        if self.capacity <= 0:
            raise ValueError("Computed capacity <= 0. Check sample_rate and window_sec.")

        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._write = 0           # write index
        self._filled = 0          # number of valid samples currently stored (<= capacity)

    def clear(self) -> None:
        self._buf.fill(0.0)
        self._write = 0
        self._filled = 0

    def __len__(self) -> int:
        """Number of valid samples in the buffer (<= capacity)."""
        return self._filled

    def append(self, frame: np.ndarray) -> None:
        """
        Append new audio samples into the rolling buffer.

        Parameters
        ----------
        frame : np.ndarray
            Mono audio samples. Accepts shape (n,) float32 (preferred) or int16.
            If int16 is provided, it is converted to float32 in [-1, 1].
        """
        if frame is None:
            return

        # Downmix if accidentally given shape (n,1) or (n,channels)
        if frame.ndim == 2:
            if frame.shape[1] == 1:
                frame = frame.reshape(-1)
            else:
                frame = frame.mean(axis=1)

        if frame.ndim != 1:
            raise ValueError("AudioRingBuffer.append expects mono 1D samples.")

        # Convert dtype
        if frame.dtype == np.int16:
            x = frame.astype(np.float32) / 32768.0
        else:
            x = frame.astype(np.float32, copy=False)

        n = x.size
        if n == 0:
            return

        # If incoming is larger than capacity, keep only the newest part
        if n >= self.capacity:
            x = x[-self.capacity:]
            n = x.size

        end = self._write + n
        if end <= self.capacity:
            self._buf[self._write:end] = x
        else:
            first = self.capacity - self._write
            self._buf[self._write:] = x[:first]
            self._buf[:end - self.capacity] = x[first:]

        self._write = (self._write + n) % self.capacity
        self._filled = min(self.capacity, self._filled + n)

    def get(self, pad_to_full: bool = False) -> np.ndarray:
        """
        Return the newest rolling window as a contiguous float32 array.

        Parameters
        ----------
        pad_to_full : bool
            - False (default): return only the valid portion (length = filled)
            - True: always return exactly `capacity` samples, left-padded with zeros if needed

        Returns
        -------
        np.ndarray (float32)
            The most recent audio in chronological order (oldest -> newest).
        """
        if self._filled == 0:
            return np.zeros(self.capacity, dtype=np.float32) if pad_to_full else np.zeros(0, dtype=np.float32)

        if pad_to_full:
            # We want a full window even if not filled; zeros will remain for older portion
            needed = self.capacity
        else:
            needed = self._filled

        start = (self._write - needed) % self.capacity

        if start < self._write and needed <= self.capacity:
            # contiguous
            out = self._buf[start:start + needed].copy()
        else:
            # wrapped
            part1 = self._buf[start:].copy()
            part2_len = needed - part1.size
            part2 = self._buf[:part2_len].copy()
            out = np.concatenate([part1, part2], axis=0)

        if pad_to_full and self._filled < self.capacity:
            # If buffer isn't full, we want left-padding (older samples) to be zeros.
            # Our current out already includes zeros only if those sections were never overwritten,
            # but early on they are zeros because we initialized to zeros and filled < capacity.
            # Still, to guarantee semantics, enforce explicitly:
            pad_len = self.capacity - self._filled
            if pad_len > 0:
                out[:pad_len] = 0.0

        return out

    def duration_sec(self) -> float:
        """How many seconds of audio are currently stored (<= window_sec)."""
        return float(self._filled) / float(self.cfg.sample_rate)


# -----------------------------
# Quick standalone sanity check
# -----------------------------
if __name__ == "__main__":
    rb = AudioRingBuffer(sample_rate=16000, window_sec=2.0)
    # Append 1 second of ones
    rb.append(np.ones(16000, dtype=np.float32))
    print("len:", len(rb), "sec:", rb.duration_sec())
    x = rb.get()
    print("get len:", x.size, "mean:", float(x.mean()))
    # Append 2 seconds of twos (will overwrite)
    rb.append(np.ones(32000, dtype=np.float32) * 2)
    y = rb.get(pad_to_full=True)
    print("after overwrite get len:", y.size, "mean:", float(y.mean()))
