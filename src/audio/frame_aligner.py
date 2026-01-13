import numpy as np
from collections import deque

class FrameAligner:
    """
    Accumulates arbitrary-length audio chunks and emits
    fixed-size frames (e.g. 320 samples).
    """

    def __init__(self, frame_size: int):
        self.frame_size = frame_size
        self.buffer = deque()

    def push(self, samples: np.ndarray):
        if samples.ndim != 1:
            raise ValueError("FrameAligner expects mono audio")

        for s in samples:
            self.buffer.append(float(s))

    def pop(self):
        if len(self.buffer) < self.frame_size:
            return None

        frame = np.array(
            [self.buffer.popleft() for _ in range(self.frame_size)],
            dtype=np.float32
        )
        return frame
