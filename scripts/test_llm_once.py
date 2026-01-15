# scripts/test_llm_once.py
import sys
sys.path.append("src")

from llm.llm_engine import LLMEngine, LLMConfig
from llm.prompt_manager import PromptManager, PromptConfig

MODEL_PATH = "./models/llm/llama-3.2-1b-instruct-q4_k_m.gguf"


def main():
    llm = LLMEngine(
        LLMConfig(
            model_path=MODEL_PATH,
            threads=8,
        )
    )

    pm = PromptManager(PromptConfig())

    messages = pm.build(
        user_text="Hello, who are you?",
        history=[]
    )

    response = llm.generate(messages)
    print("Response:\n", response)


if __name__ == "__main__":
    main()
