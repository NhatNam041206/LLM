"""
Conversation Logger for debugging and analysis.
Saves conversation text and audio recordings with timestamps.
"""
import os
import wave
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional


class ConversationLogger:
    """Logs conversations (text + audio) to files for debugging."""
    
    def __init__(
        self,
        log_dir: str = "logs/conversations",
        save_audio: bool = True,
        save_text: bool = True,
        sample_rate: int = 16000
    ):
        """
        Initialize conversation logger.
        
        Args:
            log_dir: Directory to save logs and audio files
            save_audio: Whether to save audio recordings
            save_text: Whether to save text transcripts
            sample_rate: Audio sample rate for WAV files
        """
        self.log_dir = Path(log_dir)
        self.save_audio = save_audio
        self.save_text = save_text
        self.sample_rate = sample_rate
        
        # Create timestamped session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / timestamp
        
        if save_audio:
            self.audio_dir = self.session_dir / "audio"
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            
        if save_text:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.session_dir / "conversation.log"
            self._init_log_file()
            
        self.utterance_count = 0
        
    def _init_log_file(self):
        """Initialize log file with header."""
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"=== Conversation Log ===\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 50 + "\n\n")
    
    def log_user_input(self, text: str, audio: Optional[np.ndarray] = None):
        """
        Log user input (text and optionally audio).
        
        Args:
            text: User's transcribed text
            audio: Raw audio data (float32, mono)
        """
        self.utterance_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Save text
        if self.save_text:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] USER: {text}\n")
        
        # Save audio
        if self.save_audio and audio is not None:
            audio_filename = f"user_{self.utterance_count:04d}_{timestamp.replace(':', '')}.wav"
            audio_path = self.audio_dir / audio_filename
            self._save_wav(audio_path, audio)
            
            if self.save_text:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"         Audio: {audio_filename}\n")
    
    def log_assistant_response(self, text: str):
        """
        Log assistant's response.
        
        Args:
            text: Assistant's response text
        """
        if self.save_text:
            timestamp = datetime.now().strftime("%H:%M:%S")
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] ASSISTANT: {text}\n\n")
    
    def log_system_message(self, message: str):
        """
        Log system message (e.g., mode changes, errors).
        
        Args:
            message: System message to log
        """
        if self.save_text:
            timestamp = datetime.now().strftime("%H:%M:%S")
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] SYSTEM: {message}\n\n")
    
    def _save_wav(self, filepath: Path, audio: np.ndarray):
        """
        Save audio data to WAV file.
        
        Args:
            filepath: Output WAV file path
            audio: Audio data (float32, values in [-1, 1])
        """
        # Convert float32 to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 2 bytes = 16 bits
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
    
    def close(self):
        """Close logger and write footer."""
        if self.save_text and self.log_file.exists():
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 50}\n")
                f.write(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total utterances: {self.utterance_count}\n")
        
        print(f"\n[DEBUG] Conversation saved to: {self.session_dir}")
