from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class MemoryConfig:
    max_turns: int = 6
    max_chars_per_msg: int = 500


class ConversationMemory:
    """
    Stores short conversation history:
    [{"role": "user"|"assistant", "content": "..."}]
    """

    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self._history: List[Dict[str, str]] = []

    def clear(self):
        self._history.clear()

    def add_user(self, text: str):
        self._append("user", text)

    def add_assistant(self, text: str):
        self._append("assistant", text)

    def _append(self, role: str, text: str):
        text = text.strip()
        if not text:
            return

        # trim message length
        if len(text) > self.cfg.max_chars_per_msg:
            text = text[: self.cfg.max_chars_per_msg]

        self._history.append({"role": role, "content": text})

        # keep last N turns (user+assistant = 1 turn)
        max_msgs = self.cfg.max_turns * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def get(self) -> List[Dict[str, str]]:
        return list(self._history)
