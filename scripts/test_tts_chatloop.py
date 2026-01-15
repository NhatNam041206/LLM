# scripts/test_tts_chatloop.py
import sys
sys.path.append("src")

from tts.tts_engine import PiperTTSEngine, PiperTTSConfig

MODEL = "models/tts/en_US-amy-medium.onnx"
MODEL_JSON = "models/tts/en_US-amy-medium.onnx.json"

def main():
    tts = PiperTTSEngine(
        PiperTTSConfig(
            model_path=MODEL,
            length_scale=1.0,
            noise_scale=0.5,
            noise_w_scale=0.6,
            volume=1.0,
        )
    )

    print("Type text and press Enter. Ctrl+C to exit.")
    try:
        while True:
            text = input("\nText> ").strip()
            if not text:
                continue
            tts.speak(text)
    except KeyboardInterrupt:
        tts.speak("Goodbye!")
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
