import sys
sys.path.append("src")

from tts.tts_engine import PiperTTSEngine, PiperTTSConfig

MODEL = r"D:\GW_UNIVERSITY\Hellogit\Autobot\LLM\llm\models\tts\en_US-amy-medium.onnx"  # or your voice file

def main():
    tts = PiperTTSEngine(
        PiperTTSConfig(
            model_path=MODEL,
            length_scale=1.0,
            noise_scale=0.667,
            noise_w_scale=0.8,
            volume=1.0,
        )
    )

    print("Output devices:")
    for d in tts.list_output_devices():
        print(f"[{d['index']}] {d['name']}")

    text = "Hello! This is a Piper text to speech test."
    print("Speaking:", text)
    tts.speak(text)
def debug():
    from piper.config import SynthesisConfig
    tts = PiperTTSEngine(
        PiperTTSConfig(
            model_path=MODEL,
            length_scale=1.0,
            noise_scale=0.667,
            noise_w_scale=0.8,
            volume=1.0,
        )
    )
    syn = SynthesisConfig(length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8)

    chunks = list(tts.voice.synthesize("hello test", syn_config=syn))
    print("num_chunks =", len(chunks))
    if chunks:
        c = chunks[0]
        print("chunk type:", type(c))
        print("dir chunk (sample):", [x for x in dir(c) if "audio" in x or "pcm" in x or "data" in x or "sample" in x])

if __name__ == "__main__":
    main()
    # debug()
