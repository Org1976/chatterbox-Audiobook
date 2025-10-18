"""High level validation entry points."""

from __future__ import annotations

import json
import math
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..project_management import load_project_metadata
from ..voice_management import load_voice_for_tts
from .asr_local import OfflineASR, load_asr_model, transcribe_audio
from .metrics import compute_wer
from .text_norm import normalize_text, tokenize
from ..regen.manager import regen_hints_from_reasons, regenerate_chunk_locally


DEFAULT_THRESHOLDS = {"wer": 0.25}


def _safe_duration_seconds(audio_path: Path) -> Optional[float]:
    try:
        with wave.open(str(audio_path), "rb") as handle:
            frames = handle.getnframes()
            sample_rate = handle.getframerate()
        return frames / sample_rate if sample_rate else None
    except (wave.Error, FileNotFoundError):
        return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_chunk(
    audio_path: str | Path,
    reference_text: str,
    *,
    asr: Optional[OfflineASR] = None,
    thresholds: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate a single audio chunk against the expected text."""
    audio_path = Path(audio_path)
    thresholds = thresholds or DEFAULT_THRESHOLDS
    metadata = metadata or {}

    if not audio_path.exists():
        return {
            "audio_path": str(audio_path),
            "reference_text": reference_text,
            "error": "audio_not_found",
            "passed": False,
            "timestamp": _timestamp(),
            "metadata": metadata,
        }

    if asr is None:
        asr = load_asr_model()

    transcript = transcribe_audio(audio_path, asr=asr)
    normalized_reference = normalize_text(reference_text)
    normalized_transcript = normalize_text(transcript)

    wer, alignment = compute_wer(normalized_reference, normalized_transcript)
    reference_tokens = tokenize(normalized_reference)
    transcript_tokens = tokenize(normalized_transcript)

    duration_seconds = _safe_duration_seconds(audio_path)

    result = {
        "audio_path": str(audio_path),
        "reference_text": reference_text,
        "transcript": transcript,
        "normalized_reference": normalized_reference,
        "normalized_transcript": normalized_transcript,
        "reference_token_count": len(reference_tokens),
        "transcript_token_count": len(transcript_tokens),
        "wer": float(wer),
        "passed": bool(wer <= thresholds.get("wer", DEFAULT_THRESHOLDS["wer"])),
        "alignment": [step.to_dict() for step in alignment],
        "duration_seconds": duration_seconds,
        "thresholds": thresholds,
        "timestamp": _timestamp(),
        "metadata": metadata,
    }
    return result


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio.astype(np.float32), -1.0, 1.0)
    int_samples = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(int_samples.tobytes())
    return str(path)


def _extract_chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        for key in ("text", "content", "chunk", "chunk_text"):
            if key in chunk and chunk[key]:
                return str(chunk[key])
        return ""
    return str(chunk)


def _with_defaults(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = {
        "exaggeration": 1.0,
        "temperature": 0.7,
        "cfg_weight": 1.0,
        "min_p": 0.05,
        "top_p": 1.0,
        "repetition_penalty": 1.2,
    }
    if config:
        base.update({k: v for k, v in config.items() if v is not None})
    return base


def validate_and_autofix_project(
    project_dir: str | Path,
    *,
    model: Any = None,
    voice_config: Optional[Dict[str, Any]] = None,
    audio_prompt_path: Optional[str] = None,
    thresholds: Optional[Dict[str, float]] = None,
    pause_duration: float = 0.1,
    voice_library_path: Optional[str] = None,
    reasons_map: Optional[Dict[str, List[str]]] = None,
    max_regen_attempts: int = 1,
) -> Dict[str, Any]:
    """Validate every chunk within a project and attempt deterministic autofix."""
    project_path = Path(project_dir)
    metadata = load_project_metadata(str(project_path)) or {}
    if not metadata:
        legacy_metadata_path = project_path / "project_metadata.json"
        if legacy_metadata_path.exists():
            try:
                with open(legacy_metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except Exception:
                metadata = {}
    project_name = metadata.get("project_name", project_path.name)
    chunks_data = metadata.get("chunks", [])
    chunk_texts = [_extract_chunk_text(chunk) for chunk in chunks_data]
    thresholds = thresholds or DEFAULT_THRESHOLDS

    if voice_config is None:
        voice_config = metadata.get("voice_info") or {}

    voice_config = _with_defaults(voice_config)

    if not audio_prompt_path:
        audio_prompt_path = voice_config.get("audio_file") or voice_config.get("audio_prompt_path")

    if not audio_prompt_path and voice_library_path and voice_config.get("voice_name"):
        prompt_path, loaded_config = load_voice_for_tts(voice_library_path, voice_config["voice_name"])
        if prompt_path:
            audio_prompt_path = prompt_path
            voice_config.update(_with_defaults(loaded_config))
            voice_config["audio_file"] = prompt_path

    base_pause = metadata.get("pause_duration", pause_duration)

    asr = load_asr_model()
    chunk_reports_dir = project_path / "chunks"
    chunk_reports_dir.mkdir(exist_ok=True)

    validated_chunks = 0
    best_passed = 0
    best_wer_sum = 0.0
    best_wer_entries = 0
    max_wer = 0.0
    autofixed_chunks = 0
    chunk_reports: List[Dict[str, Any]] = []

    project_basename = project_path.name
    can_regenerate = bool(model and audio_prompt_path)
    sample_rate = getattr(model, "sr", 24000) if model else 24000

    for idx, chunk_text in enumerate(chunk_texts, start=1):
        chunk_id = f"{idx:03d}"
        audio_path = project_path / f"{project_basename}_{idx:03d}.wav"
        attempts: List[Dict[str, Any]] = []
        best_attempt: Optional[Dict[str, Any]] = None
        chunk_error: Optional[str] = None

        try:
            original_result = validate_chunk(
                audio_path,
                chunk_text,
                asr=asr,
                thresholds=thresholds,
                metadata={"chunk_id": chunk_id, "attempt": "original"},
            )
            attempts.append({"attempt_type": "original", **original_result})

            if "wer" in original_result:
                validated_chunks += 1
                best_attempt = original_result
                best_wer_sum += original_result["wer"]
                best_wer_entries += 1
                if original_result["passed"]:
                    best_passed += 1
                max_wer = max(max_wer, original_result["wer"])

                should_regen = (
                    can_regenerate
                    and max_regen_attempts > 0
                    and original_result["wer"] > thresholds.get("wer", DEFAULT_THRESHOLDS["wer"])
                )

                if should_regen:
                    reasons = []
                    if reasons_map and chunk_id in reasons_map:
                        reasons = reasons_map[chunk_id]
                    if not reasons:
                        reasons = [
                            f"WER {original_result['wer']:.3f} above threshold {thresholds.get('wer', DEFAULT_THRESHOLDS['wer']):.3f}"
                        ]

                    hints = regen_hints_from_reasons(
                        reasons,
                        base_rate=float(voice_config.get("cfg_weight", 1.0)),
                        base_pause=float(base_pause),
                    )
                    regen_audio, regen_details = regenerate_chunk_locally(
                        model=model,
                        chunk_text=chunk_text,
                        voice_config=voice_config,
                        audio_prompt_path=audio_prompt_path,
                        hints=hints,
                        pause_duration=base_pause,
                    )
                    regen_path = chunk_reports_dir / f"{chunk_id}.autofix.wav"
                    saved_path = _write_wav(regen_path, regen_audio, sample_rate)
                    regen_result = validate_chunk(
                        regen_path,
                        chunk_text,
                        asr=asr,
                        thresholds=thresholds,
                        metadata={
                            "chunk_id": chunk_id,
                            "attempt": "autofix",
                            "hints": hints,
                            "saved_audio_path": saved_path,
                            "regen_details": regen_details,
                        },
                    )
                    attempts.append({"attempt_type": "autofix", **regen_result})

                    if "wer" in regen_result:
                        if regen_result["wer"] < best_attempt.get("wer", math.inf):
                            if not best_attempt.get("passed") and regen_result["passed"]:
                                best_passed += 1
                                autofixed_chunks += 1
                            elif best_attempt.get("passed") and not regen_result["passed"]:
                                best_passed -= 1
                            best_wer_sum -= best_attempt["wer"]
                            best_wer_sum += regen_result["wer"]
                            best_attempt = regen_result
                        else:
                            # Keep existing best stats but still record attempt
                            pass
                        max_wer = max(max_wer, regen_result["wer"])
            else:
                chunk_error = original_result.get("error")
        except Exception as exc:  # pragma: no cover - defensive guard
            chunk_error = str(exc)

        if best_attempt and "wer" not in best_attempt:
            best_attempt = None

        chunk_report = {
            "chunk_id": chunk_id,
            "audio_path": str(audio_path),
            "reference_text": chunk_text,
            "attempts": attempts,
            "best_attempt_type": (best_attempt and best_attempt.get("metadata", {}).get("attempt")) or None,
            "best_wer": (best_attempt or {}).get("wer"),
            "autofix_applied": any(attempt.get("metadata", {}).get("attempt") == "autofix" for attempt in attempts),
            "error": chunk_error,
        }
        chunk_reports.append(chunk_report)

        chunk_report_path = chunk_reports_dir / f"{chunk_id}.validation.json"
        with open(chunk_report_path, "w", encoding="utf-8") as handle:
            json.dump(chunk_report, handle, ensure_ascii=False, indent=2)

    average_wer = (best_wer_sum / best_wer_entries) if best_wer_entries else None

    qa_summary = {
        "project_dir": str(project_path),
        "project_name": project_name,
        "validated_at": _timestamp(),
        "total_chunks": len(chunk_texts),
        "validated_chunks": validated_chunks,
        "passed_chunks": best_passed,
        "failed_chunks": max(validated_chunks - best_passed, 0),
        "autofixed_chunks": autofixed_chunks,
        "average_wer": average_wer,
        "max_wer": max_wer if best_wer_entries else None,
        "thresholds": thresholds,
        "reports": [
            {
                "chunk_id": report["chunk_id"],
                "validation_path": str((chunk_reports_dir / f"{report['chunk_id']}.validation.json").resolve()),
                "autofix_applied": report["autofix_applied"],
                "best_wer": report["best_wer"],
            }
            for report in chunk_reports
        ],
    }

    summary_path = project_path / "qa_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(qa_summary, handle, ensure_ascii=False, indent=2)

    return qa_summary
