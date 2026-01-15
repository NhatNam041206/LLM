# src/tts/text_normalizer.py
from __future__ import annotations

import re

class TextNormalizer:
    """
    Minimal, safe normalization for TTS.
    Keep it conservative to avoid changing meaning too much.
    """

    _space_re = re.compile(r"\s+")
    _bad_chars_re = re.compile(r"[^\S\r\n]+")

    def normalize(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        # Collapse whitespace
        text = self._space_re.sub(" ", text)

        # Optional: remove weird control chars (keep punctuation)
        text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")

        # Limit crazy-long text (voice assistant shouldn’t speak novels)
        if len(text) > 1200:
            text = text[:1200].rstrip()

        return text
