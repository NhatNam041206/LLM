"""Stage C: Stability filter to avoid false triggers."""

from collections import deque
from typing import Optional
import time
from .config import StabilityConfig


class StabilityFilter:
    """Filter to require consistent face detection before triggering."""
    
    def __init__(self, config: StabilityConfig):
        """Initialize stability filter.
        
        Args:
            config: StabilityConfig with window and threshold settings
        """
        self.config = config
        self.detection_history = deque(maxlen=config.window_frames)
        self.first_detection_time: Optional[float] = None
        self.continuous_detection = False
        
    def reset(self):
        """Reset the filter state."""
        self.detection_history.clear()
        self.first_detection_time = None
        self.continuous_detection = False
        
    def update(self, face_detected: bool) -> bool:
        """Update filter with new detection result.
        
        Args:
            face_detected: Whether a face was detected in current frame
            
        Returns:
            True if stable detection confirmed (should trigger)
        """
        current_time = time.time()
        
        # Add to history
        self.detection_history.append(face_detected)
        
        # Track first detection time
        if face_detected:
            if self.first_detection_time is None:
                self.first_detection_time = current_time
        else:
            # Reset if no face detected
            self.first_detection_time = None
            self.continuous_detection = False
        
        # Not enough history yet
        if len(self.detection_history) < self.config.window_frames:
            return False
        
        # Count hits in window
        hits = sum(self.detection_history)
        
        # Check if we meet the hit threshold
        if hits >= self.config.required_hits:
            # Also check minimum duration if first detection time is set
            if self.first_detection_time is not None:
                duration = current_time - self.first_detection_time
                if duration >= self.config.min_duration_seconds:
                    self.continuous_detection = True
                    return True
        
        return False
    
    def is_stable(self) -> bool:
        """Check if currently in stable detection state.
        
        Returns:
            True if stable detection is active
        """
        return self.continuous_detection
    
    def get_hit_ratio(self) -> float:
        """Get current hit ratio in the window.
        
        Returns:
            Ratio of frames with face detection (0.0 to 1.0)
        """
        if len(self.detection_history) == 0:
            return 0.0
        return sum(self.detection_history) / len(self.detection_history)
