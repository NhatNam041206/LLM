# llm/prompt_manager.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant.\n"
    "Respond clearly and naturally.\n"
    "Keep answers short (1–3 sentences) unless the user asks for details.\n"
    "If you do not know something, say you are not sure.\n"
)


@dataclass
class PromptConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_response_chars: int = 400


class PromptManager:
    def __init__(self, cfg: PromptConfig):
        self.cfg = cfg

    def build(
        self,
        user_text: str,
        history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Returns messages in ChatML-style format
        compatible with llama.cpp chat completion.
        """
        messages: List[Dict[str, str]] = []

        # System message (fixed, trusted)
        messages.append({
            "role": "system",
            "content": self.cfg.system_prompt.strip()
        })

        # Conversation history
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Current user input
        messages.append({
            "role": "user",
            "content": user_text.strip()
        })

        return messages

    def postprocess(self, text: str) -> str:
        """
        Enforce spoken-style constraints.
        """
        text = text.strip()

        if len(text) > self.cfg.max_response_chars:
            text = text[: self.cfg.max_response_chars]

        return text
