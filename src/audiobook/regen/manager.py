"""Helpers for deterministic chunk regeneration adjustments."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from ..processing import create_silence_audio


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def regen_hints_from_reasons(
    reasons: Iterable[str],
    base_rate: float,
    base_pause: float,
) -> Dict[str, Any]:
    """Translate textual reasons into deterministic synthesis adjustments."""
    rate_multiplier = 1.0
    pause_multiplier = 1.0
    notes: List[str] = []

    for reason in sorted({reason.lower() for reason in reasons if reason}):
        if "fast" in reason:
            rate_multiplier *= 0.9
            pause_multiplier *= 1.25
            notes.append("Detected pacing complaint → slowing speech and increasing pauses")
        elif "slow" in reason or "sluggish" in reason:
            rate_multiplier *= 1.1
            pause_multiplier *= 0.85
            notes.append("Detected slow delivery → speeding up speech and shortening pauses")
        elif "pause" in reason or "breath" in reason:
            pause_multiplier *= 1.15
            notes.append("Explicit pause feedback → lengthening pauses")
        elif "clarity" in reason or "unclear" in reason:
            rate_multiplier *= 0.95
            notes.append("Clarity feedback → slightly slower speech")
        elif "staccato" in reason or "choppy" in reason:
            pause_multiplier *= 0.9
            notes.append("Choppy delivery → reducing pause length")

    adjusted_rate = _clamp(base_rate * rate_multiplier, 0.1, 3.0)
    adjusted_pause = _clamp(base_pause * pause_multiplier, 0.0, 2.0)

    return {
        "rate_multiplier": rate_multiplier,
        "pause_multiplier": pause_multiplier,
        "rate": round(adjusted_rate, 3),
        "pause_duration": round(adjusted_pause, 3),
        "notes": notes,
        "reasons": list(sorted({reason for reason in reasons if reason})),
    }


def regenerate_chunk_locally(
    *,
    model: Any,
    chunk_text: str,
    voice_config: Dict[str, Any],
    audio_prompt_path: str,
    hints: Dict[str, Any],
    pause_duration: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Regenerate a single chunk using deterministic adjustments."""
    if model is None:
        raise ValueError("Model is required for regeneration")
    if not audio_prompt_path:
        raise ValueError("Audio prompt path missing for regeneration")

    cfg_weight = float(hints.get("rate", voice_config.get("cfg_weight", 1.0)))
    exaggerated = float(voice_config.get("exaggeration", 1.0))
    temperature = float(voice_config.get("temperature", 0.7))
    min_p = float(voice_config.get("min_p", 0.05))
    top_p = float(voice_config.get("top_p", 1.0))
    repetition_penalty = float(voice_config.get("repetition_penalty", 1.2))
    pause_override = float(hints.get("pause_duration", pause_duration))

    conds = model.prepare_conditionals(audio_prompt_path, exaggerated)
    wav = model.generate(
        chunk_text,
        conds,
        exaggeration=exaggerated,
        temperature=temperature,
        cfg_weight=cfg_weight,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    if hasattr(wav, "detach"):
        audio_np = wav.detach().squeeze(0).cpu().numpy()
    elif hasattr(wav, "squeeze"):
        audio_np = wav.squeeze(0)
        if hasattr(audio_np, "cpu"):
            audio_np = audio_np.cpu().numpy()
        else:
            audio_np = np.asarray(audio_np)
    else:
        audio_np = np.asarray(wav)

    if pause_override > 0:
        sample_rate = getattr(model, "sr", 24000)
        pause_audio = create_silence_audio(pause_override, sample_rate)
        audio_np = np.concatenate([audio_np, pause_audio])

    regen_details = {
        "cfg_weight": cfg_weight,
        "pause_duration": pause_override,
        "temperature": temperature,
        "min_p": min_p,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "hints": hints,
    }
    return audio_np, regen_details
