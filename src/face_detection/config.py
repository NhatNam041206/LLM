"""Configuration for face detection pipeline."""

from dataclasses import dataclass


@dataclass
class PresenceConfig:
    """Configuration for Stage A: Presence detection."""
    window_seconds: float = 5.0  # Time window to evaluate presence
    motion_threshold: float = 500.0  # Threshold for motion detection
    presence_ratio: float = 0.3  # Fraction of frames with motion to consider "present"
    frame_skip: int = 2  # Process every Nth frame (to reduce cost)


@dataclass
class FaceDetectorConfig:
    """Configuration for Stage B: Face detection."""
    model_name: str = "haarcascade_frontalface_default"  # OpenCV cascade classifier
    min_confidence: float = 0.0  # Minimum confidence (for DNN models)
    scale_factor: float = 1.1  # Scale factor for multi-scale detection
    min_neighbors: int = 4  # Minimum neighbors for cascade
    min_face_size: tuple = (30, 30)  # Minimum face size in pixels
    use_gpu: bool = False  # Use GPU acceleration if available


@dataclass
class StabilityConfig:
    """Configuration for Stage C: Stability filter."""
    window_frames: int = 10  # Number of recent frames to consider
    required_hits: int = 7  # Minimum hits in window to trigger
    min_duration_seconds: float = 1.0  # Minimum continuous duration


@dataclass
class StateMachineConfig:
    """Configuration for state machine behavior."""
    cooldown_seconds: float = 3.0  # Cooldown after trigger
    idle_timeout_seconds: float = 30.0  # Return to IDLE if no face for this long


@dataclass
class CameraConfig:
    """Configuration for camera capture."""
    device_id: int = 0  # Camera device ID
    width: int = 640  # Frame width
    height: int = 480  # Frame height
    fps: int = 30  # Target FPS
    processing_width: int = 320  # Downscale width for processing


@dataclass
class FaceDetectionConfig:
    """Main configuration for face detection pipeline."""
    camera: CameraConfig = None
    presence: PresenceConfig = None
    face_detector: FaceDetectorConfig = None
    stability: StabilityConfig = None
    state_machine: StateMachineConfig = None
    
    def __post_init__(self):
        """Initialize sub-configs with defaults if not provided."""
        if self.camera is None:
            self.camera = CameraConfig()
        if self.presence is None:
            self.presence = PresenceConfig()
        if self.face_detector is None:
            self.face_detector = FaceDetectorConfig()
        if self.stability is None:
            self.stability = StabilityConfig()
        if self.state_machine is None:
            self.state_machine = StateMachineConfig()
