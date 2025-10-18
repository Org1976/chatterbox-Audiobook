"""Utility helpers for normalising text prior to ASR comparison."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def strip_accents(text: str) -> str:
    """Remove accent marks while preserving base characters."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    """Apply standard normalisation suitable for WER comparison."""
    if not text:
        return ""

    text = strip_accents(text.lower())
    # Replace any punctuation-like characters with spaces to avoid false positives
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return normalize_whitespace(text)


def tokenize(text: str | Iterable[str]) -> List[str]:
    """Tokenise normalised text into a sequence of words."""
    if isinstance(text, str):
        text = normalize_text(text)
        if not text:
            return []
        return _WORD_RE.findall(text)

    return [token for token in text if token]
