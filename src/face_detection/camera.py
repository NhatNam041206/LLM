"""Camera capture utility for face detection pipeline."""

import cv2
import numpy as np
from typing import Optional, Tuple
from .config import CameraConfig


class CameraCapture:
    """Camera capture with frame preprocessing."""
    
    def __init__(self, config: CameraConfig):
        """Initialize camera capture.
        
        Args:
            config: CameraConfig with camera settings
        """
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        
    def open(self) -> bool:
        """Open the camera.
        
        Returns:
            True if camera opened successfully
        """
        try:
            self.cap = cv2.VideoCapture(self.config.device_id)
            
            if not self.cap.isOpened():
                print(f"[CameraCapture] Failed to open camera {self.config.device_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            self.is_opened = True
            print(f"[CameraCapture] Camera opened: {self.config.width}x{self.config.height} @ {self.config.fps}fps")
            return True
            
        except Exception as e:
            print(f"[CameraCapture] Error opening camera: {e}")
            return False
    
    def close(self):
        """Close the camera."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print("[CameraCapture] Camera closed")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera.
        
        Returns:
            Tuple of (success: bool, frame: Optional[np.ndarray])
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, None
            return True, frame
            
        except Exception as e:
            print(f"[CameraCapture] Error reading frame: {e}")
            return False, None
    
    def read_processed(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Read and process frame for detection.
        
        Returns:
            Tuple of (success, original_frame, processed_frame)
        """
        ret, frame = self.read()
        if not ret or frame is None:
            return False, None, None
        
        # Create downscaled version for processing
        if self.config.processing_width < self.config.width:
            scale = self.config.processing_width / self.config.width
            new_height = int(self.config.height * scale)
            processed = cv2.resize(frame, (self.config.processing_width, new_height))
        else:
            processed = frame.copy()
        
        return True, frame, processed
    
    def is_ready(self) -> bool:
        """Check if camera is ready to capture.
        
        Returns:
            True if camera is opened and ready
        """
        return self.is_opened and self.cap is not None and self.cap.isOpened()
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
