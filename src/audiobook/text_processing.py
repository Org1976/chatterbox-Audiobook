"""Compatibility wrapper for text processing utilities.

This module re-exports the public helpers that used to live in
``text_processing.py``.  The implementation was moved to
``processing.py`` during a refactor, but several imports still point to
``src.audiobook.text_processing``.  Importing from here keeps those
call sites working without duplicating logic.
"""

from .processing import (
    chunk_text_by_sentences,
    adaptive_chunk_text,
    load_text_file,
    validate_audiobook_input,
    parse_multi_voice_text,
    clean_character_name_from_text,
    chunk_multi_voice_segments,
    validate_multi_voice_text,
    validate_multi_audiobook_input,
    analyze_multi_voice_text,
    _filter_problematic_short_chunks,
    save_audio_chunks,
)

__all__ = [
    "chunk_text_by_sentences",
    "adaptive_chunk_text",
    "load_text_file",
    "validate_audiobook_input",
    "parse_multi_voice_text",
    "clean_character_name_from_text",
    "chunk_multi_voice_segments",
    "validate_multi_voice_text",
    "validate_multi_audiobook_input",
    "analyze_multi_voice_text",
    "_filter_problematic_short_chunks",
    "save_audio_chunks",
]
