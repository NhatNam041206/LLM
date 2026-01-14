import sys
sys.path.append("src")

from llm.llm_engine import LLMEngine, LLMConfig
from llm.memory import ConversationMemory, MemoryConfig
from llm.prompt_manager import PromptManager, PromptConfig

MODEL_PATH = "./models/llama-3.2-1b-instruct-q4_k_m.gguf"


def main():
    llm = LLMEngine(
        LLMConfig(
            model_path=MODEL_PATH,
            threads=8,
        )
    )

    memory = ConversationMemory(
        MemoryConfig(
            max_turns=6,
            max_chars_per_msg=500,
        )
    )

    prompt_mgr = PromptManager(
        PromptConfig(
            max_response_chars=400,
        )
    )

    print("Local LLM chat (Ctrl+C to exit)\n")

    try:
        while True:
            user_text = input("You: ").strip()
            if not user_text:
                continue

            memory.add_user(user_text)

            messages = prompt_mgr.build(
                user_text=user_text,
                history=memory.get()
            )

            response = llm.generate(messages)
            response = prompt_mgr.postprocess(response)

            memory.add_assistant(response)

            print("Assistant:", response)
            print()

    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
