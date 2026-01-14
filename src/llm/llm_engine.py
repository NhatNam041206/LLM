from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict

from llama_cpp import Llama


@dataclass
class LLMConfig:
    model_path: str
    context_tokens: int = 2048
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    threads: int = 8
    gpu_layers: int = 0


class LLMEngine:
    """
    Local LLM engine using llama.cpp (GGUF).
    """

    def __init__(self, cfg: LLMConfig):
        if not os.path.exists(cfg.model_path):
            raise FileNotFoundError(f"Model not found: {cfg.model_path}")

        self.cfg = cfg

        # Load model ONCE
        self.llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.context_tokens,
            n_threads=cfg.threads,
            n_gpu_layers=cfg.gpu_layers,
            logits_all=False,
            verbose=False,
        )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response from chat messages.
        """
        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            repeat_penalty=self.cfg.repeat_penalty,
        )

        return output["choices"][0]["message"]["content"]
