"""
Test script for conversation logger functionality.
"""
import sys
import numpy as np

sys.path.append("src")

from utils.conversation_logger import ConversationLogger

def test_logger():
    print("Testing Conversation Logger...")
    
    # Create logger
    logger = ConversationLogger(
        log_dir="logs/test_session",
        save_audio=True,
        save_text=True,
        sample_rate=16000
    )
    
    # Create test audio (1 second of sine wave)
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio1 = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
    audio2 = np.sin(2 * np.pi * 880 * t).astype(np.float32) * 0.3
    
    # Log first interaction
    logger.log_user_input("Hello, how are you?", audio1)
    logger.log_assistant_response("I'm doing great! How can I help you today?")
    
    # Log second interaction
    logger.log_user_input("Tell me a joke", audio2)
    logger.log_assistant_response("Why did the programmer quit his job? Because he didn't get arrays!")
    
    # Log system message
    logger.log_system_message("Test session completed")
    
    # Close logger
    logger.close()
    
    print("\n✅ Test completed successfully!")
    print(f"Check the output in: {logger.session_dir}")
    
if __name__ == "__main__":
    test_logger()
