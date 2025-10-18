"""Metrics and alignment helpers for audiobook validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .text_norm import tokenize


@dataclass
class AlignmentStep:
    """Single step in an alignment trace."""

    operation: str
    reference: str | None
    hypothesis: str | None

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
        }


def _prepare_tokens(text: str | Sequence[str]) -> List[str]:
    if isinstance(text, (list, tuple)):
        return [token for token in text if token]
    return tokenize(text)


def _alignment_tables(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[List[List[int]], List[List[str]]]:
    rows, cols = len(ref_tokens) + 1, len(hyp_tokens) + 1
    distance = [[0] * cols for _ in range(rows)]
    backtrack = [[""] * cols for _ in range(rows)]

    for i in range(1, rows):
        distance[i][0] = i
        backtrack[i][0] = "delete"
    for j in range(1, cols):
        distance[0][j] = j
        backtrack[0][j] = "insert"

    for i in range(1, rows):
        for j in range(1, cols):
            ref_word = ref_tokens[i - 1]
            hyp_word = hyp_tokens[j - 1]

            substitution_cost = distance[i - 1][j - 1] + (0 if ref_word == hyp_word else 1)
            insertion_cost = distance[i][j - 1] + 1
            deletion_cost = distance[i - 1][j] + 1

            best_cost = substitution_cost
            operation = "correct" if ref_word == hyp_word else "substitute"

            if insertion_cost < best_cost or (insertion_cost == best_cost and operation != "correct"):
                best_cost = insertion_cost
                operation = "insert"

            if deletion_cost < best_cost:
                best_cost = deletion_cost
                operation = "delete"

            distance[i][j] = best_cost
            backtrack[i][j] = operation

    return distance, backtrack


def align_words(reference: Iterable[str], hypothesis: Iterable[str]) -> List[AlignmentStep]:
    ref_tokens = _prepare_tokens(reference)
    hyp_tokens = _prepare_tokens(hypothesis)

    if not ref_tokens and not hyp_tokens:
        return []

    distance, backtrack = _alignment_tables(ref_tokens, hyp_tokens)
    i, j = len(ref_tokens), len(hyp_tokens)
    alignment: List[AlignmentStep] = []

    while i > 0 or j > 0:
        operation = backtrack[i][j]
        if operation in {"correct", "substitute"}:
            ref_word = ref_tokens[i - 1] if i > 0 else None
            hyp_word = hyp_tokens[j - 1] if j > 0 else None
            alignment.append(AlignmentStep(operation, ref_word, hyp_word))
            i -= 1
            j -= 1
        elif operation == "delete":
            ref_word = ref_tokens[i - 1] if i > 0 else None
            alignment.append(AlignmentStep(operation, ref_word, None))
            i -= 1
        elif operation == "insert":
            hyp_word = hyp_tokens[j - 1] if j > 0 else None
            alignment.append(AlignmentStep(operation, None, hyp_word))
            j -= 1
        else:
            # Should not happen but guard against malformed backtrack table
            if i > 0:
                alignment.append(AlignmentStep("delete", ref_tokens[i - 1], None))
                i -= 1
            elif j > 0:
                alignment.append(AlignmentStep("insert", None, hyp_tokens[j - 1]))
                j -= 1

    alignment.reverse()
    return alignment


def compute_wer(reference: str | Sequence[str], hypothesis: str | Sequence[str]) -> Tuple[float, List[AlignmentStep]]:
    ref_tokens = _prepare_tokens(reference)
    hyp_tokens = _prepare_tokens(hypothesis)

    if not ref_tokens:
        return (0.0 if not hyp_tokens else 1.0, align_words(ref_tokens, hyp_tokens))

    alignment = align_words(ref_tokens, hyp_tokens)
    errors = sum(1 for step in alignment if step.operation != "correct")
    wer = errors / max(1, len(ref_tokens))
    return wer, alignment
