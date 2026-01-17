"""State machine for face detection pipeline."""

from enum import Enum, auto
import time
from typing import Optional, Callable
from .config import StateMachineConfig


class State(Enum):
    """States for the face detection state machine."""
    IDLE = auto()           # Only presence detection runs
    FACE_MODE = auto()      # Face detection + stability check active
    COOLDOWN = auto()       # Triggered, waiting before resetting


class StateMachine:
    """State machine to control pipeline behavior."""
    
    def __init__(self, config: StateMachineConfig):
        """Initialize state machine.
        
        Args:
            config: StateMachineConfig with timing settings
        """
        self.config = config
        self.state = State.IDLE
        self.state_entry_time = time.time()
        self.last_face_time: Optional[float] = None
        
    def reset(self):
        """Reset state machine to IDLE."""
        self.state = State.IDLE
        self.state_entry_time = time.time()
        self.last_face_time = None
        
    def get_state(self) -> State:
        """Get current state."""
        return self.state
    
    def is_idle(self) -> bool:
        """Check if in IDLE state."""
        return self.state == State.IDLE
    
    def is_face_mode(self) -> bool:
        """Check if in FACE_MODE state."""
        return self.state == State.FACE_MODE
    
    def is_cooldown(self) -> bool:
        """Check if in COOLDOWN state."""
        return self.state == State.COOLDOWN
    
    def _transition_to(self, new_state: State):
        """Transition to a new state."""
        if new_state != self.state:
            print(f"[StateMachine] {self.state.name} -> {new_state.name}")
            self.state = new_state
            self.state_entry_time = time.time()
    
    def update(self, presence_ok: bool, face_detected: bool, trigger: bool) -> State:
        """Update state machine based on current conditions.
        
        Args:
            presence_ok: Result from Stage A (presence detection)
            face_detected: Result from Stage B (face detection)
            trigger: Result from Stage C (stability filter)
            
        Returns:
            Current state after update
        """
        current_time = time.time()
        
        # Track last face detection time
        if face_detected:
            self.last_face_time = current_time
        
        # State transitions
        if self.state == State.IDLE:
            # Enter FACE_MODE if presence detected
            if presence_ok:
                self._transition_to(State.FACE_MODE)
                
        elif self.state == State.FACE_MODE:
            # Trigger event if stable face detection
            if trigger:
                self._transition_to(State.COOLDOWN)
            # Return to IDLE if no face for too long
            elif self.last_face_time is not None:
                time_since_face = current_time - self.last_face_time
                if time_since_face > self.config.idle_timeout_seconds:
                    self._transition_to(State.IDLE)
                    
        elif self.state == State.COOLDOWN:
            # Exit cooldown after timeout
            time_in_cooldown = current_time - self.state_entry_time
            if time_in_cooldown >= self.config.cooldown_seconds:
                self._transition_to(State.IDLE)
        
        return self.state
    
    def should_run_face_detection(self) -> bool:
        """Check if face detection should run in current state.
        
        Returns:
            True if face detection should be active
        """
        return self.state == State.FACE_MODE
    
    def should_emit_event(self) -> bool:
        """Check if we just entered cooldown (event should be emitted).
        
        Returns:
            True if we just triggered (entered cooldown this update)
        """
        # Check if we just transitioned to cooldown
        return self.state == State.COOLDOWN and \
               (time.time() - self.state_entry_time) < 0.1  # Within 100ms of transition


class EventEmitter:
    """Emits events when face detection triggers."""
    
    def __init__(self, callback: Optional[Callable[[], None]] = None):
        """Initialize event emitter.
        
        Args:
            callback: Optional callback function to call on trigger
        """
        self.callback = callback
        self.trigger_count = 0
        
    def emit(self):
        """Emit a trigger event."""
        self.trigger_count += 1
        print(f"[EventEmitter] 🎯 TRIGGER #{self.trigger_count}: Face detected and stable!")
        
        if self.callback is not None:
            try:
                self.callback()
            except Exception as e:
                print(f"[EventEmitter] Error in callback: {e}")
    
    def get_trigger_count(self) -> int:
        """Get total number of triggers emitted."""
        return self.trigger_count
