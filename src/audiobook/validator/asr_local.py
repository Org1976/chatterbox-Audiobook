"""Offline ASR helpers built on torchaudio pipelines."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torchaudio


@dataclass
class OfflineASR:
    """Container for a lightweight offline ASR model."""

    model: torch.nn.Module
    labels: list[str]
    sample_rate: int
    device: str = "cpu"

    def decode(self, emissions: torch.Tensor) -> str:
        """Decode emissions using greedy CTC decoding."""
        tokens = torch.argmax(emissions, dim=-1)
        transcript = []
        prev_token = None
        labels = self.labels
        for token in tokens.squeeze(0).tolist():
            if token == prev_token:
                continue
            prev_token = token
            if token == 0:
                continue
            piece = labels[token]
            transcript.append(piece)
        return "".join(transcript).replace("|", " ").strip()


@functools.lru_cache(maxsize=2)
def load_asr_model(device: str = "cpu") -> OfflineASR:
    """Load and cache the offline ASR pipeline."""
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    model.eval()
    labels = bundle.get_labels()
    return OfflineASR(model=model, labels=labels, sample_rate=bundle.sample_rate, device=device)


def _prepare_waveform(audio_path: Path, target_sample_rate: int, device: str) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform.to(device)


def transcribe_audio(audio_path: str | Path, asr: Optional[OfflineASR] = None) -> str:
    """Transcribe an audio file using the offline ASR model."""
    audio_path = Path(audio_path)
    if asr is None:
        asr = load_asr_model()

    waveform = _prepare_waveform(audio_path, asr.sample_rate, asr.device)
    with torch.inference_mode():
        emissions, _ = asr.model(waveform)
    return asr.decode(emissions.cpu())
