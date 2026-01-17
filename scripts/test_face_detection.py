"""Test script for face detection pipeline."""

import sys
import cv2

sys.path.append("src")

from face_detection import FaceDetectionPipeline, FaceDetectionConfig
from face_detection.config import CameraConfig, PresenceConfig, FaceDetectorConfig, StabilityConfig, StateMachineConfig


def test_face_detection_basic():
    """Test basic face detection without orchestrator."""
    print("="*60)
    print("Face Detection Test")
    print("="*60)
    print("This will run the face detection pipeline.")
    print("Position yourself in front of the camera.")
    print("Press Ctrl+C to stop.\n")
    
    # Create config with defaults
    config = FaceDetectionConfig()
    
    # You can customize settings here:
    # config.presence.motion_threshold = 1000.0
    # config.stability.required_hits = 5
    
    # Callback when face detected
    trigger_count = 0
    def on_trigger():
        nonlocal trigger_count
        trigger_count += 1
        print(f"\n{'='*60}")
        print(f"✅ TRIGGER #{trigger_count}: Face detected and stable!")
        print(f"{'='*60}\n")
    
    # Create and run pipeline
    pipeline = FaceDetectionPipeline(config, on_trigger=on_trigger)
    
    try:
        pipeline.run(visualize=True)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        # Print statistics
        stats = pipeline.get_stats()
        print("\n" + "="*60)
        print("Statistics:")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Presence hits: {stats['presence_hits']}")
        print(f"  Face hits: {stats['face_hits']}")
        print(f"  Triggers: {stats['triggers']}")
        print("="*60)


def test_face_detection_single_trigger():
    """Test waiting for a single trigger."""
    print("="*60)
    print("Single Trigger Test")
    print("="*60)
    print("Waiting for one face detection trigger...")
    print("Position yourself in front of camera.\n")
    
    config = FaceDetectionConfig()
    
    pipeline = FaceDetectionPipeline(config)
    
    # Wait for trigger with 30 second timeout
    triggered = pipeline.wait_for_trigger(timeout=30.0)
    
    if triggered:
        print("\n✅ Successfully detected face!")
    else:
        print("\n❌ Timeout - no face detected")
    
    stats = pipeline.get_stats()
    print(f"\nStats: {stats['triggers']} trigger(s), {stats['frames_processed']} frames")


def test_camera_only():
    """Test just camera capture."""
    print("="*60)
    print("Camera Test")
    print("="*60)
    print("Testing camera capture only.")
    print("Press 'q' to quit.\n")
    
    from face_detection.camera import CameraCapture
    
    camera_cfg = CameraConfig(device_id=0, width=640, height=480)
    
    with CameraCapture(camera_cfg) as camera:
        print("Camera opened successfully!")
        
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to read frame")
                break
            
            cv2.imshow("Camera Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    print("Camera test complete")


def test_face_detector_only():
    """Test face detector on camera feed."""
    print("="*60)
    print("Face Detector Test")
    print("="*60)
    print("Testing face detection only (no pipeline).")
    print("Press 'q' to quit.\n")
    
    from face_detection.camera import CameraCapture
    from face_detection.face_detector import FaceDetector
    
    camera_cfg = CameraConfig(device_id=0)
    detector_cfg = FaceDetectorConfig()
    
    camera = CameraCapture(camera_cfg)
    camera.open()
    
    detector = FaceDetector(detector_cfg)
    
    face_count = 0
    frame_count = 0
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces
            has_face, detections = detector.detect(frame)
            
            if has_face:
                face_count += 1
                # Draw bounding boxes
                for det in detections:
                    x, y, w, h = det.bbox
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display stats
            cv2.putText(frame, f"Frames: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Face hits: {face_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if has_face else (255, 255, 255), 2)
            
            cv2.imshow("Face Detector Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()
    
    print(f"\nDetected faces in {face_count}/{frame_count} frames ({100*face_count/frame_count:.1f}%)")


def main():
    """Run tests based on command line argument."""
    if len(sys.argv) < 2:
        print("Usage: python test_face_detection.py <test_name>")
        print("\nAvailable tests:")
        print("  camera         - Test camera capture only")
        print("  detector       - Test face detector with bounding boxes")
        print("  pipeline       - Test full pipeline with visualization")
        print("  single         - Wait for single trigger")
        print()
        print("Example: python scripts/test_face_detection.py pipeline")
        return
    
    test_name = sys.argv[1].lower()
    
    if test_name == "camera":
        test_camera_only()
    elif test_name == "detector":
        test_face_detector_only()
    elif test_name == "pipeline":
        test_face_detection_basic()
    elif test_name == "single":
        test_face_detection_single_trigger()
    else:
        print(f"Unknown test: {test_name}")
        print("Available: camera, detector, pipeline, single")


if __name__ == "__main__":
    main()
