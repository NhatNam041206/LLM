"""Stage B: Face detection using OpenCV."""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from .config import FaceDetectorConfig


@dataclass
class FaceDetection:
    """Result of face detection."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float = 1.0


class FaceDetector:
    """Face detection using OpenCV Haar Cascades or DNN models."""
    
    def __init__(self, config: FaceDetectorConfig):
        """Initialize face detector.
        
        Args:
            config: FaceDetectorConfig with model settings
        """
        self.config = config
        self.detector = None
        self._load_detector()
        
    def _load_detector(self):
        """Load the face detection model."""
        try:
            # Use Haar Cascade (lightweight, CPU-friendly)
            cascade_path = cv2.data.haarcascades + f"{self.config.model_name}.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
            
            if self.detector.empty():
                raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")
                
        except Exception as e:
            print(f"[FaceDetector] Error loading detector: {e}")
            # Fallback to frontal face default
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
            
            if self.detector.empty():
                raise RuntimeError("Failed to load default face cascade")
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, List[FaceDetection]]:
        """Detect faces in a frame.
        
        Args:
            frame: Input frame (BGR or grayscale)
            
        Returns:
            Tuple of (face_detected: bool, detections: List[FaceDetection])
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Detect faces using Haar Cascade
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=self.config.min_face_size
        )
        
        # Convert to FaceDetection objects
        detections = []
        for (x, y, w, h) in faces:
            detections.append(FaceDetection(
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=1.0  # Haar cascades don't provide confidence scores
            ))
        
        return len(detections) > 0, detections
    
    def detect_simple(self, frame: np.ndarray) -> bool:
        """Simplified detection returning only boolean result.
        
        Args:
            frame: Input frame
            
        Returns:
            True if at least one face detected
        """
        has_face, _ = self.detect(frame)
        return has_face


class DNNFaceDetector(FaceDetector):
    """Face detection using DNN-based models (more accurate but heavier)."""
    
    def __init__(self, config: FaceDetectorConfig):
        """Initialize DNN face detector.
        
        Args:
            config: FaceDetectorConfig with model settings
        """
        self.config = config
        self.net = None
        self._load_dnn_model()
        
    def _load_dnn_model(self):
        """Load DNN model (e.g., Caffe-based face detector)."""
        # This is a placeholder for DNN-based detection
        # You can use models like:
        # - OpenCV's DNN face detector
        # - MediaPipe Face Detection
        # - MTCNN
        # For now, fall back to Haar Cascade
        super().__init__(self.config)
        
    def detect(self, frame: np.ndarray) -> Tuple[bool, List[FaceDetection]]:
        """Detect faces using DNN model."""
        # Placeholder - implement DNN detection here if needed
        # For now, use parent Haar Cascade implementation
        return super().detect(frame)
