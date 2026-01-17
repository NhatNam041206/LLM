"""Stage A: Presence detection using motion/background subtraction."""

import cv2
import numpy as np
from collections import deque
from typing import Optional
from .config import PresenceConfig


class PresenceDetector:
    """Lightweight presence detection using frame differencing."""
    
    def __init__(self, config: PresenceConfig):
        """Initialize presence detector.
        
        Args:
            config: PresenceConfig with threshold and window settings
        """
        self.config = config
        self.motion_history = deque(maxlen=int(config.window_seconds * 30 / config.frame_skip))
        self.prev_frame = None
        self.frame_count = 0
        
    def reset(self):
        """Reset the detector state."""
        self.motion_history.clear()
        self.prev_frame = None
        self.frame_count = 0
        
    def detect_motion(self, frame: np.ndarray) -> bool:
        """Detect motion in a single frame.
        
        Args:
            frame: Input frame (BGR or grayscale)
            
        Returns:
            True if motion detected above threshold
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # First frame - no motion
        if self.prev_frame is None:
            self.prev_frame = gray
            return False
            
        # Compute absolute difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        
        # Compute motion score (sum of differences)
        motion_score = np.sum(frame_diff)
        
        # Update previous frame
        self.prev_frame = gray
        
        # Return True if motion exceeds threshold
        return motion_score > self.config.motion_threshold
        
    def check_presence(self, frame: np.ndarray) -> bool:
        """Check for presence based on motion history.
        
        Args:
            frame: Input frame
            
        Returns:
            True if presence detected (motion in enough frames)
        """
        self.frame_count += 1
        
        # Skip frames to reduce processing
        if self.frame_count % self.config.frame_skip != 0:
            return len(self.motion_history) > 0 and \
                   sum(self.motion_history) / len(self.motion_history) >= self.config.presence_ratio
        
        # Detect motion in this frame
        has_motion = self.detect_motion(frame)
        self.motion_history.append(has_motion)
        
        # Not enough history yet
        if len(self.motion_history) < 3:
            return False
            
        # Check if motion present in enough frames
        motion_ratio = sum(self.motion_history) / len(self.motion_history)
        return motion_ratio >= self.config.presence_ratio
