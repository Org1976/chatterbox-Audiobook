"""Quality assurance utilities for audiobook regeneration workflows."""

from __future__ import annotations

import copy
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        raise ModuleNotFoundError("tomllib or tomli is required to load QA configuration") from exc

from .audio_processing import analyze_audio_quality

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QA_CONFIG_PATH = REPO_ROOT / "config" / "qa.defaults.toml"


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries without mutating inputs."""
    result: Dict[str, Any] = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result.get(key, {}), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_default_qa_config() -> Dict[str, Any]:
    """Load the repository-wide default QA configuration."""
    if not DEFAULT_QA_CONFIG_PATH.exists():
        return {"qa": {}}

    with DEFAULT_QA_CONFIG_PATH.open("rb") as handle:
        return tomllib.load(handle)


def ensure_project_qa_file(project_dir: Path | str) -> Path:
    """Ensure a project-level QA configuration exists by copying defaults if needed."""
    project_path = Path(project_dir)
    project_path.mkdir(parents=True, exist_ok=True)

    qa_path = project_path / "qa.toml"
    if qa_path.exists():
        return qa_path

    if DEFAULT_QA_CONFIG_PATH.exists():
        shutil.copy2(DEFAULT_QA_CONFIG_PATH, qa_path)
    else:  # Fallback in case defaults are missing
        qa_path.write_text("[qa]\n", encoding="utf-8")
    return qa_path


def load_project_qa_config(project_dir: Optional[Path | str]) -> Dict[str, Any]:
    """Load QA configuration for a project, merged with defaults."""
    defaults = load_default_qa_config()
    config = copy.deepcopy(defaults)

    if not project_dir:
        return config

    qa_path = Path(project_dir) / "qa.toml"
    if not qa_path.exists():
        return config

    with qa_path.open("rb") as handle:
        project_config = tomllib.load(handle)

    return _deep_merge(config, project_config)


def get_retry_settings(qa_config: Dict[str, Any]) -> Dict[str, float]:
    """Extract retry configuration with sensible defaults."""
    retry_cfg = qa_config.get("qa", {}).get("retry", {})
    return {
        "generation_max_attempts": int(retry_cfg.get("generation_max_attempts", 3)),
        "validation_max_attempts": int(retry_cfg.get("validation_max_attempts", 1)),
        "cooldown_seconds": float(retry_cfg.get("cooldown_seconds", 0.0)),
    }


def _to_db(value: Optional[float]) -> float:
    if value is None or value <= 0:
        return float("-inf")
    return 20.0 * math.log10(value)


def _ratio_to_db(value: Optional[float]) -> Optional[float]:
    if value is None or value <= 0:
        return None
    return 20.0 * math.log10(value)


def validate_audio_file(file_path: Path | str, qa_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a regenerated audio file against QA thresholds."""
    result: Dict[str, Any] = {
        "passed": True,
        "issues": [],
        "warnings": [],
        "metrics": {},
        "thresholds": qa_config.get("qa", {}).get("thresholds", {}),
    }

    metrics = analyze_audio_quality(str(file_path))
    result["metrics"] = metrics

    if "error" in metrics:
        result["passed"] = False
        result["issues"].append(f"Quality analysis failed: {metrics['error']}")
        return result

    thresholds = qa_config.get("qa", {}).get("thresholds", {})

    duration = metrics.get("duration")
    rms_db = _to_db(metrics.get("rms_level"))
    peak_db = _to_db(metrics.get("peak_level"))
    dynamic_range_db = _ratio_to_db(metrics.get("dynamic_range"))
    zcr = metrics.get("zero_crossing_rate")

    result["metrics"].update({
        "rms_db": rms_db,
        "peak_db": peak_db,
        "dynamic_range_db": dynamic_range_db,
        "zero_crossing_rate": zcr,
    })

    min_duration = thresholds.get("min_duration")
    if min_duration is not None and duration is not None and duration < float(min_duration):
        result["passed"] = False
        result["issues"].append(
            f"Duration {duration:.2f}s shorter than minimum {float(min_duration):.2f}s"
        )

    max_duration = thresholds.get("max_duration")
    if max_duration is not None and duration is not None and duration > float(max_duration):
        result["passed"] = False
        result["issues"].append(
            f"Duration {duration:.2f}s exceeds maximum {float(max_duration):.2f}s"
        )

    min_rms_db = thresholds.get("min_rms_db")
    if min_rms_db is not None and rms_db != float("-inf") and rms_db < float(min_rms_db):
        result["passed"] = False
        result["issues"].append(
            f"Average loudness {rms_db:.1f} dB below minimum {float(min_rms_db):.1f} dB"
        )

    max_rms_db = thresholds.get("max_rms_db")
    if max_rms_db is not None and rms_db != float("-inf") and rms_db > float(max_rms_db):
        result["passed"] = False
        result["issues"].append(
            f"Average loudness {rms_db:.1f} dB above maximum {float(max_rms_db):.1f} dB"
        )

    min_peak_db = thresholds.get("min_peak_db")
    if min_peak_db is not None and peak_db != float("-inf") and peak_db < float(min_peak_db):
        result["passed"] = False
        result["issues"].append(
            f"Peak level {peak_db:.1f} dB below minimum {float(min_peak_db):.1f} dB"
        )

    max_peak_db = thresholds.get("max_peak_db")
    if max_peak_db is not None and peak_db != float("-inf") and peak_db > float(max_peak_db):
        result["passed"] = False
        result["issues"].append(
            f"Peak level {peak_db:.1f} dB above maximum {float(max_peak_db):.1f} dB"
        )

    local_settings = qa_config.get("qa", {}).get("local", {})
    validator_dir = local_settings.get("validator_model_dir")
    validator_available = False
    if validator_dir:
        validator_path = Path(validator_dir).expanduser()
        if validator_path.exists():
            validator_available = True
        else:
            result["warnings"].append(
                f"Validator model directory not found: {validator_path}"
            )
    else:
        result["warnings"].append("Validator model directory not configured; advanced QA checks skipped.")

    if validator_available:
        max_dynamic_range_db = thresholds.get("max_dynamic_range_db")
        if (
            max_dynamic_range_db is not None
            and dynamic_range_db is not None
            and dynamic_range_db > float(max_dynamic_range_db)
        ):
            result["passed"] = False
            result["issues"].append(
                f"Dynamic range {dynamic_range_db:.1f} dB above maximum {float(max_dynamic_range_db):.1f} dB"
            )

        min_zcr = thresholds.get("min_zero_crossing_rate")
        if min_zcr is not None and zcr is not None and zcr < float(min_zcr):
            result["passed"] = False
            result["issues"].append(
                f"Zero-crossing rate {zcr:.3f} below minimum {float(min_zcr):.3f}"
            )

        max_zcr = thresholds.get("max_zero_crossing_rate")
        if max_zcr is not None and zcr is not None and zcr > float(max_zcr):
            result["passed"] = False
            result["issues"].append(
                f"Zero-crossing rate {zcr:.3f} above maximum {float(max_zcr):.3f}"
            )
    else:
        for key in ("max_dynamic_range_db", "min_zero_crossing_rate", "max_zero_crossing_rate"):
            if key in thresholds:
                result["warnings"].append(
                    f"Advanced QA threshold '{key}' skipped because validator model is unavailable."
                )

    for key in ("asr_model_dir", "alignment_model_dir"):
        configured_path = local_settings.get(key)
        if configured_path:
            resolved = Path(configured_path).expanduser()
            if not resolved.exists():
                result["warnings"].append(f"Local path for '{key}' not found: {resolved}")

    return result
