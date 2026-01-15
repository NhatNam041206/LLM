# INSTRUCTIONS — LLM Process 
**(Local Chat LLM for Voice Assistant)**

Goal
----
Provide a small, local LLM module to produce short spoken replies from final STT text.

Scope
-----
- This repository implements a minimal `LLMEngine` using `llama-cpp-python` (`Llama`) and a `PromptManager` that builds Chat-style messages.
- The LLM is used after STT produces a final utterance; the orchestrator should pass messages (history + new user text) to the engine and receive a single reply string.

Core API (implemented)
-----------------------
- `LLMEngine(LLMConfig).generate(messages: List[Dict]) -> str` — returns assistant reply.
	- `messages` should be ChatML-style list of dicts: `[{'role':'system','content':...}, {'role':'user','content':...}, ...]`.

Config (see `src/llm/llm_engine.py`)
----------------------------------
- `model_path` (str): path to GGUF model or Llama-compatible model file.
- `context_tokens` (int): context length (n_ctx).
- `max_tokens` (int): generation length limit.
- `temperature`, `top_p`, `repeat_penalty`, `threads`, `gpu_layers` — standard LLM tuning knobs.

Prompting (see `src/llm/prompt_manager.py`)
-----------------------------------------
- `PromptManager.build(user_text, history)` constructs messages with a fixed system prompt first, followed by history then the user message.
- Keep system prompt short and authoritative; the default enforces short spoken responses.
- `PromptManager.postprocess(text)` truncates very long replies to a configured max character length.

Best practices for voice assistant replies
----------------------------------------
- Reply short (1–3 sentences) unless the user asks for details.
- Avoid long code blocks, lists, or multi-step instructions spoken at length.
- If uncertain, answer conservatively ("I’m not sure") and offer to look up or ask clarification.

Integration notes
-----------------
- Build messages via `PromptManager.build()` then call `LLMEngine.generate()`.
- Keep history trimmed to last 4–8 turns to stay within token limits on small models.

Example (pseudo)
-----------------
```
from llm.prompt_manager import PromptManager, PromptConfig
from llm.llm_engine import LLMEngine, LLMConfig

pm = PromptManager(PromptConfig())
llm = LLMEngine(LLMConfig(model_path='models/gguf/model.gguf'))

history = [{'role':'user','content':'Hi'} , {'role':'assistant','content':'Hello!'}]
messages = pm.build('What is the weather like?', history)
reply = llm.generate(messages)
reply = pm.postprocess(reply)
```

Notes
-----
- For low-RAM setups, use small GGUF models and limit `context_tokens` and `max_tokens`.
- Consider running a local inference server or using quantized models for reliability on tiny machines.