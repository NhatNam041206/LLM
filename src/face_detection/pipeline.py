"""Main face detection pipeline integrating all stages."""

import cv2
import time
from typing import Optional, Callable
from .config import FaceDetectionConfig
from .camera import CameraCapture
from .presence_detector import PresenceDetector
from .face_detector import FaceDetector
from .stability_filter import StabilityFilter
from .state_machine import StateMachine, EventEmitter, State


class FaceDetectionPipeline:
    """Complete face detection pipeline with multi-stage filtering."""
    
    def __init__(
        self,
        config: Optional[FaceDetectionConfig] = None,
        on_trigger: Optional[Callable[[], None]] = None
    ):
        """Initialize face detection pipeline.
        
        Args:
            config: FaceDetectionConfig with all stage settings
            on_trigger: Optional callback when face is detected and stable
        """
        self.config = config or FaceDetectionConfig()
        
        # Initialize all components
        self.camera = CameraCapture(self.config.camera)
        self.presence_detector = PresenceDetector(self.config.presence)
        self.face_detector = FaceDetector(self.config.face_detector)
        self.stability_filter = StabilityFilter(self.config.stability)
        self.state_machine = StateMachine(self.config.state_machine)
        self.event_emitter = EventEmitter(callback=on_trigger)
        
        # Runtime state
        self.running = False
        self.frame_count = 0
        self.last_trigger_time: Optional[float] = None
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "presence_hits": 0,
            "face_hits": 0,
            "triggers": 0
        }
    
    def start(self) -> bool:
        """Start the pipeline.
        
        Returns:
            True if started successfully
        """
        print("[FaceDetectionPipeline] Starting...")
        
        if not self.camera.open():
            print("[FaceDetectionPipeline] Failed to open camera")
            return False
        
        self.running = True
        print("[FaceDetectionPipeline] Pipeline started")
        return True
    
    def stop(self):
        """Stop the pipeline."""
        print("[FaceDetectionPipeline] Stopping...")
        self.running = False
        self.camera.close()
        print("[FaceDetectionPipeline] Pipeline stopped")
    
    def reset(self):
        """Reset all components to initial state."""
        self.presence_detector.reset()
        self.stability_filter.reset()
        self.state_machine.reset()
        self.frame_count = 0
        print("[FaceDetectionPipeline] Reset complete")
    
    def process_frame(self) -> tuple[bool, State]:
        """Process a single frame through the pipeline.
        
        Returns:
            Tuple of (triggered: bool, current_state: State)
        """
        # Read frame
        ret, frame_orig, frame_processed = self.camera.read_processed()
        if not ret or frame_processed is None:
            return False, self.state_machine.get_state()
        
        self.frame_count += 1
        self.stats["frames_processed"] += 1
        
        # Stage A: Presence detection (always runs)
        presence_ok = self.presence_detector.check_presence(frame_processed)
        if presence_ok:
            self.stats["presence_hits"] += 1
        
        # Stage B: Face detection (only if in FACE_MODE)
        face_detected = False
        if self.state_machine.should_run_face_detection():
            face_detected = self.face_detector.detect_simple(frame_processed)
            if face_detected:
                self.stats["face_hits"] += 1
        
        # Stage C: Stability filter (only if face detected)
        trigger = False
        if face_detected:
            trigger = self.stability_filter.update(face_detected)
        else:
            # Update with False to maintain history
            self.stability_filter.update(False)
        
        # Update state machine
        self.state_machine.update(presence_ok, face_detected, trigger)
        
        # Check if we should emit event
        if self.state_machine.should_emit_event():
            self.event_emitter.emit()
            self.last_trigger_time = time.time()
            self.stats["triggers"] += 1
            return True, self.state_machine.get_state()
        
        return False, self.state_machine.get_state()
    
    def run(self, visualize: bool = False):
        """Run the pipeline in a loop.
        
        Args:
            visualize: If True, display camera feed with detection overlays
        """
        if not self.start():
            return
        
        print("[FaceDetectionPipeline] Running... Press 'q' to quit")
        
        try:
            while self.running:
                triggered, state = self.process_frame()
                
                # Optional: visualize
                if visualize:
                    self._visualize_status(state)
                
                # Small delay to avoid maxing out CPU
                cv2.waitKey(1)
                
        except KeyboardInterrupt:
            print("\n[FaceDetectionPipeline] Interrupted by user")
        finally:
            self.stop()
            if visualize:
                cv2.destroyAllWindows()
    
    def _visualize_status(self, state: State):
        """Display status information (for debugging/demo).
        
        Args:
            state: Current state machine state
        """
        # Read current frame
        ret, frame = self.camera.read()
        if not ret or frame is None:
            return
        
        # Draw state and stats
        cv2.putText(frame, f"State: {state.name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Triggers: {self.stats['triggers']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Face Detection Pipeline", frame)
    
    def get_stats(self) -> dict:
        """Get pipeline statistics.
        
        Returns:
            Dictionary with pipeline statistics
        """
        return self.stats.copy()
    
    def wait_for_trigger(self, timeout: Optional[float] = None) -> bool:
        """Wait for a single trigger event.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            True if triggered, False if timeout
        """
        if not self.start():
            return False
        
        start_time = time.time()
        initial_triggers = self.stats["triggers"]
        
        try:
            while self.running:
                triggered, _ = self.process_frame()
                
                # Check if we got a new trigger
                if self.stats["triggers"] > initial_triggers:
                    return True
                
                # Check timeout
                if timeout is not None:
                    if (time.time() - start_time) >= timeout:
                        print(f"[FaceDetectionPipeline] Timeout after {timeout}s")
                        return False
                
                time.sleep(0.01)  # Small delay
                
        except KeyboardInterrupt:
            print("\n[FaceDetectionPipeline] Interrupted by user")
            return False
        finally:
            self.stop()
        
        return False
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
